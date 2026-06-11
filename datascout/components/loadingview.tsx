/**
 * datascout.components.LoadingView
 * Animated pipeline progress — exposes system intelligence during wait
 */

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useStore } from '../../lib/store'

const PIPELINE_STAGES = [
  { id: 'parse',      label: 'Query Parse',       desc: 'Detecting task type, domain, and search intent',        icon: 'ti-brain' },
  { id: 'retrieval',  label: 'Multi-source Fetch', desc: 'Parallel queries to HuggingFace, OpenML, Kaggle',       icon: 'ti-database' },
  { id: 'evaluate',   label: 'Evaluator Pipeline', desc: '5-dimension deterministic scoring per dataset',          icon: 'ti-math-function' },
  { id: 'rank',       label: 'Deterministic Rank', desc: 'Diversity-boosted ranking by composite score',           icon: 'ti-sort-descending' },
  { id: 'explain',    label: 'AI Explanation',     desc: 'LLM generates human-readable insights (never rankings)', icon: 'ti-sparkles' },
]

const LOADING_STEPS = [
  'Parsing query intent and task type…',
  'Querying HuggingFace Datasets Hub…',
  'Querying OpenML dataset repository…',
  'Querying Kaggle competition datasets…',
  'Running deterministic evaluator pipeline…',
  'Scoring relevance, quality, freshness, popularity…',
  'Applying diversity boost to ranking…',
  'Generating research insights…',
]

export function LoadingView() {
  const { sessions, activeSessionId } = useStore()
  const query = sessions.find(s => s.id === activeSessionId)?.query ?? ''

  const [currentStep, setCurrentStep] = useState(0)
  const [completedStages, setCompletedStages] = useState<string[]>([])
  const [activeStage, setActiveStage] = useState(0)

  useEffect(() => {
    // Progress through loading steps
    const stepTimer = setInterval(() => {
      setCurrentStep(prev => Math.min(prev + 1, LOADING_STEPS.length - 1))
    }, 700)

    // Progress through pipeline stages
    const stageTimer = setInterval(() => {
      setActiveStage(prev => {
        const next = Math.min(prev + 1, PIPELINE_STAGES.length - 1)
        if (prev < PIPELINE_STAGES.length - 1) {
          setCompletedStages(c => [...c, PIPELINE_STAGES[prev].id])
        }
        return next
      })
    }, 1200)

    return () => {
      clearInterval(stepTimer)
      clearInterval(stageTimer)
    }
  }, [])

  return (
    <div style={{ maxWidth: '740px', margin: '0 auto', padding: '8px 0' }}>
      {/* Query echo */}
      <div style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: '12px',
        padding: '16px 20px',
        marginBottom: '20px',
      }}>
        <div style={{ fontSize: '10px', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '4px', fontWeight: 600 }}>
          Researching
        </div>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: '20px', color: 'var(--text)', letterSpacing: '-0.3px' }}>
          {query}
        </div>
      </div>

      {/* Pipeline diagram */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 0,
        marginBottom: '28px',
        overflowX: 'auto',
        padding: '4px 0',
      }}>
        {PIPELINE_STAGES.map((stage, i) => {
          const isDone = completedStages.includes(stage.id)
          const isActive = i === activeStage && !isDone
          const isPending = !isDone && !isActive

          const bgColor = isDone
            ? 'rgba(34,197,94,0.1)'
            : isActive
            ? 'rgba(79,125,255,0.12)'
            : 'var(--bg)'
          const borderColor = isDone
            ? 'rgba(34,197,94,0.35)'
            : isActive
            ? 'rgba(79,125,255,0.4)'
            : 'var(--border)'
          const textColor = isDone
            ? 'var(--green)'
            : isActive
            ? 'var(--accent)'
            : 'var(--text3)'

          return (
            <div key={stage.id} style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
              <motion.div
                animate={{ scale: isActive ? [1, 1.03, 1] : 1 }}
                transition={{ repeat: isActive ? Infinity : 0, duration: 1.5 }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                <div style={{
                  padding: '6px 12px',
                  borderRadius: '8px',
                  background: bgColor,
                  border: `1px solid ${borderColor}`,
                  color: textColor,
                  fontSize: '11px',
                  fontFamily: 'var(--font-mono)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  whiteSpace: 'nowrap',
                  transition: 'all 0.3s',
                }}>
                  <i
                    className={`ti ${isDone ? 'ti-check' : isActive ? 'ti-loader-2' : stage.icon} ${isActive ? 'animate-spin' : ''}`}
                    aria-hidden="true"
                    style={{ fontSize: '13px' }}
                  />
                  {stage.label}
                </div>
                <div style={{
                  fontSize: '9px',
                  color: isDone ? 'var(--green)' : isActive ? 'var(--accent)' : 'var(--text3)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.4px',
                  transition: 'color 0.3s',
                }}>
                  {isDone ? 'complete' : isActive ? 'running' : 'pending'}
                </div>
              </motion.div>
              {i < PIPELINE_STAGES.length - 1 && (
                <div style={{
                  width: '28px',
                  height: '1px',
                  background: isDone ? 'rgba(34,197,94,0.4)' : 'var(--border)',
                  margin: '0 4px',
                  marginBottom: '16px',
                  transition: 'background 0.3s',
                  flexShrink: 0,
                }} />
              )}
            </div>
          )
        })}
      </div>

      {/* Stage detail card */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeStage}
          initial={{ opacity: 0, x: 8 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -8 }}
          transition={{ duration: 0.2 }}
          style={{
            background: 'rgba(79,125,255,0.06)',
            border: '1px solid rgba(79,125,255,0.2)',
            borderRadius: '10px',
            padding: '14px 18px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '14px',
          }}
        >
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'rgba(79,125,255,0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}>
            <i
              className={`ti ${PIPELINE_STAGES[activeStage]?.icon ?? 'ti-loader-2'} animate-spin`}
              aria-hidden="true"
              style={{ fontSize: '18px', color: 'var(--accent)' }}
            />
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text)', marginBottom: '3px' }}>
              {PIPELINE_STAGES[activeStage]?.label}
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text2)', lineHeight: 1.5 }}>
              {PIPELINE_STAGES[activeStage]?.desc}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Live step ticker */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '10px 14px',
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: '8px',
      }}>
        <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent)', animation: 'pulse 1.5s ease-in-out infinite', flexShrink: 0 }} />
        <AnimatePresence mode="wait">
          <motion.span
            key={currentStep}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.2 }}
            style={{ fontSize: '12px', color: 'var(--text2)', fontFamily: 'var(--font-mono)' }}
          >
            {LOADING_STEPS[currentStep]}
          </motion.span>
        </AnimatePresence>
      </div>

      {/* System note */}
      <div style={{
        marginTop: '20px',
        fontSize: '11px',
        color: 'var(--text3)',
        textAlign: 'center',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '5px',
      }}>
        <i className="ti ti-lock" aria-hidden="true" style={{ fontSize: '12px' }} />
        Ranking is computed deterministically — LLM is only used to explain results
      </div>
    </div>
  )
}