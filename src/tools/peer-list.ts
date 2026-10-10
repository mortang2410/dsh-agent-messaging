/**
 * `peer_list` — which sessions this one can address.
 */

import { defineTool, type ToolDefinition } from '@deepseek-ai/dsh-tools'

import type { CardStore } from '../adapters/cards.ts'
import type { WorkspaceFacts } from '../domain/derived-card.ts'
import { deriveCard } from '../domain/derived-card.ts'
import type { TaskStateStore } from '../adapters/task-states.ts'
import type { WorkClaims } from '../app/claim-work.ts'
import type { MessageSender, SenderIdentity } from '../app/send-message.ts'
import { summarizeCard } from '../domain/card.ts'
import { describeAge, peerAddress, type PeerDescriptor } from '../domain/peer.ts'
import { summarizeTaskState } from '../domain/task-state.ts'
import { presentListCall } from './presentation.ts'
import { requireCallerSessionId } from './caller.ts'

/**
 * Reachability as the model should reason about it.
 *
 * A session that is not running carries its age, because the benchmark caught a
 * session deferring work to peers that had been dead for hours: it read their
 * titles, concluded the job was theirs, and left a call site broken. A stopped
 * session and a colleague looked identical in the listing, and a title is a
 * strong, unreliable signal.
 * @param peer - the peer to describe.
 * @param now - the current time, supplied so a listing is testable.
 * @returns the state, with an age for anything that is not running.
 */
function reachability(peer: PeerDescriptor, now: number): string {
  switch (peer.location.kind) {
    case 'local':
      return peer.status === 'running' ? 'running' : 'idle'
    case 'remote':
      return 'running (another host process)'
    case 'a2a':
      return 'external agent (A2A)'
    case 'offline':
      return `not running, started ${describeAge(Math.max(0, now - peer.createdAt))}`
  }
}

/** One row of a `peer_list` result: the calling session or one peer. */
interface ListedSession {
  readonly name: string
  readonly session_id: string
  readonly state: string
  readonly is_self: boolean
  readonly title?: string
  readonly cwd?: string
  readonly task?: string
  readonly role?: string
  working_on?: string[]
}

/**
 * Render one `peer_list` row as the block a model reads.
 * @param entry - the row to render.
 * @returns a SELF or PEER block with the full session id on its own line.
 */
function renderEntry(entry: ListedSession): string {
  const label = entry.is_self ? 'SELF' : 'PEER'
  const title = entry.title ? ` "${entry.title}"` : ''
  const lines = [`${label}: ${entry.name} [${entry.state}]${title}`, `  session_id: ${entry.session_id}`]
  if (entry.cwd) lines.push(`  cwd: ${entry.cwd}`)
  if (entry.task) lines.push(`  task: ${entry.task}`)
  if (entry.role) lines.push(`  ${entry.role}`)
  if (entry.working_on && entry.working_on.length > 0) {
    lines.push(`  working on: ${entry.working_on.join(', ')}`)
  }
  return lines.join('\n')
}

/**
 * Build the `peer_list` tool.
 * @param sender - the send use case, which also owns discovery.
 * @param claims - live work claims, so a caller sees who is on what.
 * @param cards - capability cards, so a caller sees what each peer is for.
 * @param states - declared task states, so a caller sees what each peer's work is doing.
 * @param identify - resolves the calling session's own name and directory, from its authoritative id.
 * @param describe - reads a workspace for an honest card where none was published.
 * @returns the registry-ready definition.
 */
export function createPeerListTool(
  sender: MessageSender,
  claims: WorkClaims,
  cards: CardStore,
  states: TaskStateStore,
  identify: (sessionId: string, signal?: AbortSignal) => Promise<SenderIdentity>,
  describe?: (cwd: string) => Promise<Omit<WorkspaceFacts, 'sessionId'>>,
): ToolDefinition {
  return defineTool({
    name: 'peer_list',
    description: [
      'List the agent sessions you can address with peer_send. The first entry, marked SELF, is this session:',
      'its session_id is your own canonical ID, and other sessions see you under that same ID as a PEER.',
      'You cannot send to SELF. Each entry shows the full session_id. Use this listing when you need the exact',
      'session ID of a peer, or when an address was ambiguous.',
      'Sessions marked "not running" can still be sent to:',
      'the message is queued and delivered when that session next starts.',
      'This returns only session identities — never their conversation contents.',
      'A session marked "not running" is not working on anything: do not conclude that a task belongs to one',
      'because its title mentions that task. If work needs doing and nobody live owns it, it is yours or your user\'s.',
    ].join(' '),
    parameters: {},
    output: {
      schema: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            name: { type: 'string', required: true, description: 'The address to pass to peer_send.' },
            session_id: {
              type: 'string',
              required: true,
              description: 'The full, canonical session ID. Exact and unambiguous, unlike a name.',
            },
            state: { type: 'string', required: true, description: 'running, idle, or not running.' },
            is_self: {
              type: 'boolean',
              required: true,
              description: 'True for the calling session itself, which is always the first entry. It cannot be messaged.',
            },
            title: { type: 'string' },
            cwd: { type: 'string' },
            role: {
              type: 'string',
              description: "What this peer says it is for, and what it owns. Its own words, not a guess from its title.",
            },
            task: {
              type: 'string',
              description:
                "What this peer's work is doing, in its own words: working, blocked, done or abandoned. More informative than the process state.",
            },
            working_on: {
              type: 'array',
              description: 'Resources this peer has claimed. Do not edit these without talking to them.',
              items: { type: 'string' },
            },
          },
        },
      },
      render: (_args, value) => {
        if (value.length === 0) return [{ type: 'text', text: 'No sessions are reachable.' }]
        const blocks = value.map(renderEntry)
        // The calling session is always listed, so "no peers" is its only other case.
        if (value.every((entry) => entry.is_self)) blocks.push('No other sessions are reachable.')
        return [{ type: 'text', text: blocks.join('\n\n') }]
      },
    },
    presentCall: () => presentListCall(),
    async execute(_args, exec) {
      // The caller is read from the execution, never from arguments: a model
      // cannot list itself as someone else, or ask for a SELF it does not own.
      const self = requireCallerSessionId(exec)
      const [identity, peers, held, published, declared] = await Promise.all([
        identify(self, exec.signal),
        sender.peers(self, exec.signal),
        claims.all(),
        cards.readAll(),
        states.readAll(),
      ])
      const cardBySession = new Map(published.map((card) => [card.sessionId, card]))

      // A session is useful to its peers before it declares anything: where a
      // card was never published, the workspace is read for the honest part of
      // one, marked as inferred so nobody mistakes it for a statement.
      if (describe !== undefined) {
        const now = Date.now()
        const candidates = [
          { sessionId: self, cwd: identity.cwd },
          ...peers.map((peer) => ({ sessionId: peer.sessionId, cwd: peer.cwd })),
        ]
        await Promise.all(
          candidates
            .filter((entry) => !cardBySession.has(entry.sessionId) && entry.cwd !== undefined)
            .map(async (entry) => {
              try {
                const facts = await describe(entry.cwd as string)
                const card = deriveCard({ sessionId: entry.sessionId, ...facts }, now)
                if (card !== undefined) cardBySession.set(entry.sessionId, card)
              } catch {
                // A listing must survive an unreadable directory.
              }
            }),
        )
      }
      // A wait is recorded as a session id; a reader needs the address. The
      // calling session is included, so a peer waiting on it reads its name.
      const addressBySession = new Map<string, string>([
        [self, identity.name],
        ...peers.map((peer): [string, string] => [peer.sessionId, peerAddress(peer)]),
      ])
      const stateBySession = new Map(declared.map((state) => [state.sessionId, state]))

      /** Claimed resources by holding session, so a caller sees who is on what. */
      const byHolder = new Map<string, string[]>()
      for (const claim of held) {
        const list = byHolder.get(claim.sessionId) ?? []
        list.push(`${claim.resource} (${claim.intent})`)
        byHolder.set(claim.sessionId, list)
      }

      /** The role, task and claims a session has, shared by the SELF and PEER rows. */
      const metadataFor = (sessionId: string) => {
        const card = cardBySession.get(sessionId)
        const task = stateBySession.get(sessionId)
        const working = byHolder.get(sessionId) ?? []
        return {
          ...(task === undefined
            ? {}
            : { task: summarizeTaskState(task, (id) => addressBySession.get(id)) }),
          ...(card === undefined ? {} : { role: summarizeCard(card) }),
          ...(working.length === 0 ? {} : { working_on: working }),
        }
      }

      const selfEntry: ListedSession = {
        name: identity.name,
        session_id: self,
        // The caller is, by definition, running: it is the session making this call.
        state: 'running',
        is_self: true,
        ...(identity.cwd === undefined ? {} : { cwd: identity.cwd }),
        ...metadataFor(self),
      }

      const now = Date.now()
      const peerEntries = peers.map(
        (peer): ListedSession => ({
          name: peerAddress(peer),
          session_id: peer.sessionId,
          state: reachability(peer, now),
          is_self: false,
          ...(peer.title === undefined ? {} : { title: peer.title }),
          ...(peer.cwd === undefined ? {} : { cwd: peer.cwd }),
          ...metadataFor(peer.sessionId),
        }),
      )

      return [selfEntry, ...peerEntries]
    },
  })
}
