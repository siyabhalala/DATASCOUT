/**
 * datascout.lib.store
 * ─────────────────────────────────────────────────────────
 * Zustand store — single source of truth for workspace state.
 * Persists sessions to localStorage with version key.
 *
 * Design:
 *   - Sessions are immutable once stored (append-only log)
 *   - Active session drives workspace view
 *   - Compare selection is ephemeral (not persisted)
 */

import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { DatasetCard, ResearchSession, SearchMeta, ResearchInsights, SystemHealth } from '../types'

interface WorkspaceStore {
  // ── Sessions ────────────────────────────────────────────
  sessions: ResearchSession[]
  activeSessionId: string | null

  // ── Results (derived from active session) ───────────────
  results: DatasetCard[]
  meta: SearchMeta | null
  insights: ResearchInsights | null

  // ── UI State ────────────────────────────────────────────
  view: 'welcome' | 'loading' | 'results' | 'error'
  loadingStep: string
  errorMessage: string

  // ── Compare ─────────────────────────────────────────────
  compareIds: string[]  // max 3

  // ── System health ────────────────────────────────────────
  health: SystemHealth | null

  // ── Actions ──────────────────────────────────────────────
  setView: (view: WorkspaceStore['view'], step?: string) => void
  setError: (msg: string) => void
  setHealth: (h: SystemHealth) => void

  startSession: (query: string) => string  // returns new session id
  completeSession: (
    id: string,
    results: DatasetCard[],
    meta: SearchMeta,
    insights: ResearchInsights
  ) => void
  activateSession: (id: string) => void
  deleteSession: (id: string) => void
  togglePinSession: (id: string) => void
  newSession: () => void

  toggleCompare: (id: string) => void
  clearCompare: () => void
}

export const useStore = create<WorkspaceStore>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      results: [],
      meta: null,
      insights: null,
      view: 'welcome',
      loadingStep: '',
      errorMessage: '',
      compareIds: [],
      health: null,

      setView: (view, step = '') => set({ view, loadingStep: step }),
      setError: (errorMessage) => set({ view: 'error', errorMessage }),
      setHealth: (health) => set({ health }),

      startSession: (query) => {
        const id = `s-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
        const session: ResearchSession = {
          id,
          query,
          created_at: new Date().toISOString(),
          result_count: 0,
          pinned: false,
          saved: false,
        }
        set(s => ({
          sessions: [session, ...s.sessions].slice(0, 50), // keep last 50
          activeSessionId: id,
          results: [],
          meta: null,
          insights: null,
          view: 'loading',
          compareIds: [],
        }))
        return id
      },

      completeSession: (id, results, meta, insights) => {
        set(s => ({
          sessions: s.sessions.map(sess =>
            sess.id === id
              ? { ...sess, result_count: results.length, results, meta, insights }
              : sess
          ),
          results,
          meta,
          insights,
          view: 'results',
        }))
      },

      activateSession: (id) => {
        const session = get().sessions.find(s => s.id === id)
        if (!session) return
        set({
          activeSessionId: id,
          results: session.results ?? [],
          meta: session.meta ?? null,
          insights: session.insights ?? null,
          view: session.results?.length ? 'results' : 'welcome',
          compareIds: [],
        })
      },

      deleteSession: (id) => {
        const { activeSessionId } = get()
        set(s => ({
          sessions: s.sessions.filter(sess => sess.id !== id),
          activeSessionId: activeSessionId === id ? null : activeSessionId,
          view: activeSessionId === id ? 'welcome' : s.view,
        }))
      },

      togglePinSession: (id) => {
        set(s => ({
          sessions: s.sessions.map(sess =>
            sess.id === id ? { ...sess, pinned: !sess.pinned } : sess
          ),
        }))
      },

      newSession: () => {
        set({
          activeSessionId: null,
          results: [],
          meta: null,
          insights: null,
          view: 'welcome',
          compareIds: [],
        })
      },

      toggleCompare: (id) => {
        const { compareIds } = get()
        if (compareIds.includes(id)) {
          set({ compareIds: compareIds.filter(x => x !== id) })
        } else if (compareIds.length < 3) {
          set({ compareIds: [...compareIds, id] })
        }
      },

      clearCompare: () => set({ compareIds: [] }),
    }),
    {
      name: 'datascout-workspace-v5',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({
        // Only persist sessions — UI state is ephemeral
        sessions: s.sessions.map(({ results: _, ...sess }) => sess), // strip large result payloads
      }),
    }
  )
)