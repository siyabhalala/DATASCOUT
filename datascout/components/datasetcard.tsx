/**
 * datascout.components.DatasetCard
 * ─────────────────────────────────────────────────────────
 * Core UI unit — exposes evaluator intelligence visually.
 *
 * Design intent:
 *   - Score is front-and-center (composite, not LLM-generated)
 *   - 5 evaluator dimensions always visible in collapsed state
 *   - Expanded: why-ranked, strengths, weaknesses, score breakdown
 *   - Never hides evaluator intelligence behind clicks
 */

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import clsx from 'clsx'
import type { DatasetCard as DatasetCardType } from '../../types'
import { useStore } from '../../lib/store'
import { ScorePill } from '../ui/ScorePill'
import { MetricBar } from '../ui/MetricBar'

const SOURCE_CONFIG: Record<string, { label: string; cls: string }> = {
  huggingface:   { label: 'HuggingFace', cls: 'src-hf' },
  hugging_face:  { label: 'HuggingFace', cls: 'src-hf' },
  openml:        { label: 'OpenML',      cls: 'src-openml' },
  kaggle:        { label: 'Kaggle',      cls: 'src-kaggle' },
}

const RANK_STYLE: Record<number, string> = {
  1: 'rank-gold',
  2: 'rank-silver',
  3: 'rank-bronze',
}

const DIMS = [
  { key: 'task_relevance' as const,    label: 'Relevance',    color: '#4f7dff' },
  { key: 'quality' as const,           label: 'Quality',      color: '#22c55e' },
  { key: 'freshness' as const,         label: 'Freshness',    color: '#00c9a7' },
  { key: 'popularity' as const,        label: 'Popularity',   color: '#f59e0b' },
  { key: 'description_match' as const, label: 'Desc Match',   color: '#a78bfa' },
]

interface DatasetCardProps {
  dataset: DatasetCardType
}

export function DatasetCard({ dataset: d }: DatasetCardProps) {
  const [expanded, setExpanded] = useState(false)
  const { compareIds, toggleCompare } = useStore()
  const isInCompare = compareIds.includes(d.id)
  const canAddCompare = compareIds.length < 3 || isInCompare

  const src = SOURCE_CONFIG[d.source] ?? { label: d.source, cls: 'src-openml' }
  const rankCls = RANK_STYLE[d.rank] ?? 'rank-n'
  const score = d.scores.composite

  return (
    <motion.div
      layout
      className={clsx('ds-card', expanded && 'ds-card-expanded')}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: d.rank * 0.04 }}
    >
      {/* ── Top row ── */}
      <div className="ds-card-top" onClick={() => setExpanded(e => !e)} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && setExpanded(x => !x)}>
        <div className={clsx('rank-badge', rankCls)}>#{d.rank}</div>

        <div className="ds-card-info">
          <div className="ds-title">{d.title}</div>
          <div className="ds-source-row">
            <span className={clsx('source-badge', src.cls)}>{src.label}</span>
            {d.task_types.slice(0, 2).map(t => (
              <span key={t} className="task-type-tag">{t}</span>
            ))}
            {d.metadata.gated && (
              <span className="gated-badge">
                <i className="ti ti-lock" aria-hidden="true" /> Gated
              </span>
            )}
          </div>
          <p className="ds-desc">{d.description || 'No description available.'}</p>
          <div className="ds-tags">
            {d.tags.slice(0, 5).map(tag => (
              <span key={tag} className="ds-tag">{tag}</span>
            ))}
          </div>
        </div>

        {/* Score column */}
        <div className="ds-card-right">
          <ScorePill value={score} />
          <button
            className={clsx('compare-toggle', isInCompare && 'compare-toggle-active')}
            onClick={e => { e.stopPropagation(); if (canAddCompare) toggleCompare(d.id) }}
            disabled={!canAddCompare && !isInCompare}
            title={isInCompare ? 'Remove from compare' : 'Add to compare'}
          >
            {isInCompare ? '✓ Comparing' : '+ Compare'}
          </button>
          <button className="expand-btn" aria-label={expanded ? 'Collapse' : 'Expand'}>
            <i className={clsx('ti', expanded ? 'ti-chevron-up' : 'ti-chevron-down')} aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* ── Metrics row (always visible) ── */}
      <div className="ds-metrics">
        {DIMS.map(dim => (
          <MetricBar
            key={dim.key}
            label={dim.label}
            value={d.scores[dim.key]}
            color={dim.color}
          />
        ))}
      </div>

      {/* ── Expanded content ── */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            className="ds-expanded"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {/* Why ranked */}
            {d.explanation.why_ranked && (
              <div className="why-box">
                <div className="why-box-title">
                  <i className="ti ti-sparkles" aria-hidden="true" />
                  Why This Ranked #{d.rank}
                </div>
                <p className="why-box-text">{d.explanation.why_ranked}</p>
              </div>
            )}

            {/* Strengths / Weaknesses */}
            <div className="sw-grid">
              <div className="sw-section">
                <div className="sw-title sw-strengths-title">
                  <i className="ti ti-check" aria-hidden="true" />
                  Strengths
                </div>
                <ul className="sw-list">
                  {d.explanation.strengths.map((s, i) => (
                    <li key={i} className="sw-strength">{s}</li>
                  ))}
                </ul>
              </div>
              <div className="sw-section">
                <div className="sw-title sw-weaknesses-title">
                  <i className="ti ti-alert-triangle" aria-hidden="true" />
                  Weaknesses
                </div>
                <ul className="sw-list">
                  {d.explanation.weaknesses.map((w, i) => (
                    <li key={i} className="sw-weakness">{w}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Score breakdown (evaluator transparency) */}
            <div className="score-breakdown">
              <div className="label" style={{ marginBottom: '10px' }}>Evaluator Score Breakdown</div>
              <div className="breakdown-grid">
                {DIMS.map(dim => (
                  <div key={dim.key} className="breakdown-dim">
                    <div className="breakdown-label">{dim.label}</div>
                    <div
                      className="breakdown-value"
                      style={{ color: dim.color }}
                    >
                      {Math.round(d.scores[dim.key] * 100)}
                    </div>
                    <div className="breakdown-bar">
                      <div
                        className="breakdown-fill"
                        style={{
                          width: `${Math.round(d.scores[dim.key] * 100)}%`,
                          background: dim.color,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Metadata */}
            <div className="ds-metadata-row">
              {d.metadata.downloads != null && (
                <span className="meta-pill">
                  <i className="ti ti-download" aria-hidden="true" />
                  {d.metadata.downloads.toLocaleString()} downloads
                </span>
              )}
              {d.metadata.license && (
                <span className="meta-pill">
                  <i className="ti ti-license" aria-hidden="true" />
                  {d.metadata.license}
                </span>
              )}
              {d.updated_at && (
                <span className="meta-pill">
                  <i className="ti ti-calendar" aria-hidden="true" />
                  Updated {d.updated_at.slice(0, 10)}
                </span>
              )}
              {d.metadata.row_count != null && (
                <span className="meta-pill">
                  <i className="ti ti-table" aria-hidden="true" />
                  {d.metadata.row_count.toLocaleString()} rows
                </span>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}