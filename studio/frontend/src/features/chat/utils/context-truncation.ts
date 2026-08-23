// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { OpenAIChatChunk } from "../types/api";

export type ContextTruncation = NonNullable<
  OpenAIChatChunk["context_truncated"]
>;

function spreadSum(
  key: "archived_messages" | "recalled_chunks",
  a: number | undefined,
  b: number | undefined,
): Record<string, number> {
  if (a === undefined && b === undefined) return {};
  return { [key]: (a ?? 0) + (b ?? 0) };
}

/**
 * Whether the fit removed turns from the prompt that was actually sent.
 *
 * Not `fits`: a fit under the physical window that misses the reply reserve still sends
 * the shortened prompt with `fits: false`, and those turns are just as gone. Every path
 * that returned the ORIGINAL messages reports `dropped_messages` as zero.
 */
export function promptWasShortened(
  truncation: ContextTruncation | undefined,
): truncation is ContextTruncation {
  return (truncation?.dropped_messages ?? 0) > 0;
}

export function compactionBoundary(
  truncation: ContextTruncation | undefined,
): number {
  if (!promptWasShortened(truncation)) return 0;
  // boundary_messages is where the boundary sits in the saved transcript, and every fit
  // that evicts records one, rescues included. So the fallback is only for turns saved
  // before it existed, hence gated on `fits`, the one shape those turns have: elsewhere
  // dropped_messages is a per-refit total, not a position, and reading it as one sets a
  // high-water mark `showsNotice` never sees exceeded again.
  return (
    truncation.boundary_messages ??
    (truncation.fits ? (truncation.dropped_messages ?? 0) : 0)
  );
}

function nonNegativeInt(value: number | undefined): number {
  // A field that arrived as null, NaN or a float is a field we cannot subtract with, and
  // silently propagating NaN would print "NaN tokens on its own".
  return Number.isFinite(value) ? Math.max(0, Math.trunc(value as number)) : 0;
}

export function latestTurnOwnTokens(
  truncation: ContextTruncation | null | undefined,
): number {
  // `latest_turn_tokens` prices a whole rendered PROMPT, not a bare message: the
  // template's wrapper and, on a tool-enabled request, the entire tool catalogue
  // rendered into the system turn. So the catalogue sits inside this number AND inside
  // `irreducible_tokens`, where it does not cancel -- the built-in catalogue alone is
  // over a thousand tokens and enabled MCP tools are added uncapped, so a 20-token
  // "hi" was reported as thousands of tokens "on its own" and blamed for a prompt the
  // catalogue filled. `shared_prompt_tokens` is that floor, measured by the server on
  // an empty prompt; taking it off leaves what the turn itself contributed.
  const latest = nonNegativeInt(truncation?.latest_turn_tokens);
  // Never the whole turn: a floor at or above the number it belongs to does not
  // describe it, and a turn reported as zero tokens is a worse lie than the old one.
  const shared = Math.min(
    nonNegativeInt(truncation?.shared_prompt_tokens),
    Math.max(0, latest - 1),
  );
  return latest - shared;
}

export function latestTurnIsTheProblem(
  truncation: ContextTruncation | null | undefined,
  budget: number,
): boolean {
  if (!truncation) return false;
  // `latest_turn_exact: false` means nothing could price the turn at all, so the number
  // is the message's own JSON at four characters a token while every other number here
  // is a tokenizer count of a rendered prompt. The two do not share units: 16,400
  // characters of newlines estimate 8,207 against 557 rendered. Quoting it as "N tokens
  // on its own" states a number the turn never cost. A turn the template renders as
  // nothing on its own is NOT this case -- the server prices that by difference and
  // reports it exact. Absent means a server that predates the flag, which only ever
  // sent a count.
  if (!(truncation.latest_turn_exact ?? true)) return false;
  // The turn WITHOUT the shared floor, so a turn that clears the budget only once a
  // tool catalogue is standing beside it is not called a turn the budget cannot hold.
  // An older server sends no floor, which reads as zero and leaves this exactly as it
  // was before the field existed.
  return latestTurnOwnTokens(truncation) > budget;
}

export function historyCannotHelp(
  truncation: ContextTruncation | null | undefined,
): boolean {
  if (!truncation) return false;
  // `irreducible_tokens` is what the fit measured AFTER dropping every group the evictor
  // was willing to drop, so it prices the floor eviction cannot go below: the template
  // wrapper, the tool catalogue, every system turn and the newest turn. At or over the
  // WINDOW, llama-server refuses on size alone however short the conversation gets, so
  // "start a new chat" is not merely vague, it opens a chat that fails identically.
  // Below the window, shortening really can work -- the fit refuses at `prompt_target`
  // but passes the untrimmed messages on -- so that case keeps the generic advice.
  const irreducible = nonNegativeInt(truncation.irreducible_tokens);
  const window = nonNegativeInt(truncation.context_length);
  return irreducible > 0 && window > 0 && irreducible >= window;
}

export function mergeContextTruncation(
  current: ContextTruncation | undefined,
  incoming: ContextTruncation,
): ContextTruncation {
  if (!current) return incoming;

  const merged = {
    ...current,
    ...incoming,
    dropped_messages: current.dropped_messages + incoming.dropped_messages,
    prompt_tokens_before:
      current.prompt_tokens_before ?? incoming.prompt_tokens_before,
    prompt_tokens_after:
      incoming.prompt_tokens_after ?? current.prompt_tokens_after,
    // A turn can compact more than once (the tool loop refits per iteration), so these
    // accumulate rather than taking the last chunk's value. Spread conditionally so a
    // plain rolling-window response keeps its old shape, with no archive keys set to
    // undefined.
    ...spreadSum("archived_messages", current.archived_messages, incoming.archived_messages),
    ...spreadSum("recalled_chunks", current.recalled_chunks, incoming.recalled_chunks),
  };

  // boundary_messages needs no rule: it is absolute, so the spread above already keeps
  // the latest fit's value. Summing it is the bug it exists to fix. boundary_anchor rides
  // along with it for the same reason, and the two must come from the SAME fit.

  // The irreducible diagnosis describes ONE fit that gave up, so an earlier failure
  // followed by a later success would otherwise leave those numbers on a result that fit.
  // Deleted rather than spread as undefined, which would put both keys on every ordinary
  // response; delete on an absent key is a no-op.
  if (incoming.fits) {
    delete merged.irreducible_tokens;
    delete merged.latest_turn_tokens;
    // Rides with the count it describes: alone it says nothing, and left behind it would
    // describe a number that is no longer there.
    delete merged.latest_turn_exact;
    // Likewise the floor: a stale one subtracted from a later fit's count would move the
    // blame rather than remove it.
    delete merged.shared_prompt_tokens;
  }
  return merged;
}
