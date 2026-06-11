'use client'

import { useState, useRef, useEffect } from 'react'
import {
  searchDatasets, SearchResponse, DatasetResult, SearchSession,
  BiasSignal, PerClassStat, NullColumn, ColumnDetail, DeepAnalysis,
} from '../lib/api'
import Image from 'next/image'

/* ── CONSTANTS ────────────────────────────────────────────────── */
const EXAMPLES = [
  "I'm building a crop disease detector for Indian farms",
  "Need sentiment analysis data for finance tweets",
  "Looking for diabetes prediction datasets with patient records",
  "Time series data for electricity demand forecasting",
  "Image datasets for detecting potholes on roads",
  "Speech recognition training data in Hindi or Gujarati",
]

const SOURCE_META: Record<string, { label: string; color: string; bg: string; border: string }> = {
  huggingface: { label: 'HuggingFace', color: '#C8A882', bg: '#FEF9F4', border: '#F0E4D4' },
  openml:      { label: 'OpenML',      color: '#7B8FA6', bg: '#F4F5F8', border: '#CDD2DC' },
  kaggle:      { label: 'Kaggle',      color: '#7A9E8A', bg: '#F2F7F4', border: '#C8DDD2' },
}

/* ── LOGO COMPONENT ───────────────────────────────────────────── */
// Renders your startup logo if /public/logo.png exists, falls back to DS monogram.
function Logo({ size = 36, radius = 10 }: { size?: number; radius?: number }) {
  const [imgError, setImgError] = useState(false)
  if (imgError) {
    return (
      <div style={{
        width: size, height: size, borderRadius: radius,
        background: '#0F0D0C', display: 'flex', alignItems: 'center',
        justifyContent: 'center', flexShrink: 0,
      }}>
        <span style={{ fontFamily: "'Jost',sans-serif", fontSize: size * 0.33, fontWeight: 600, color: '#FAFAFA' }}>DS</span>
      </div>
    )
  }
  return (
    <div style={{ width: size, height: size, borderRadius: radius, overflow: 'hidden', flexShrink: 0, background: '#F8F4F0', border: '1px solid #EAE4E2' }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/logo.png"
        alt="DataScout"
        width={size}
        height={size}
        onError={() => setImgError(true)}
        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
      />
    </div>
  )
}

/* ── PROGRESS TRACKER ─────────────────────────────────────────── */

type ProgressStage =
  | 'IDLE'
  | 'SEARCHING'         // searching all three sources
  | 'KAGGLE_DONE'
  | 'HF_DONE'
  | 'OPENML_DONE'
  | 'DEDUPLICATING'
  | 'METADATA_RANKING'
  | 'LIGHT_ANALYSIS'
  | 'RERANKING'
  | 'DEEP_ANALYSIS'
  | 'INSIGHTS'
  | 'COMPLETE'

interface ProgressStep {
  id: string
  label: string
  sublabel?: string
}

const PIPELINE_STEPS: ProgressStep[] = [
  { id: 'search',    label: 'Searching sources',      sublabel: 'Kaggle · HuggingFace · OpenML' },
  { id: 'dedup',     label: 'Deduplicating results'  },
  { id: 'metadata',  label: 'Ranking by metadata'    },
  { id: 'light',     label: 'Running light analysis' },
  { id: 'rerank',    label: 'Re-ranking candidates'  },
  { id: 'deep',      label: 'Running deep analysis'  },
  { id: 'insights',  label: 'Generating insights'    },
  { id: 'final',     label: 'Preparing results'      },
]

const STAGE_TO_STEP_INDEX: Record<string, number> = {
  SEARCHING:       0,
  KAGGLE_DONE:     0,
  HF_DONE:         0,
  OPENML_DONE:     0,
  DEDUPLICATING:   1,
  METADATA_RANKING:2,
  LIGHT_ANALYSIS:  3,
  RERANKING:       4,
  DEEP_ANALYSIS:   5,
  INSIGHTS:        6,
  COMPLETE:        7,
}

function StepIcon({ state }: { state: 'done' | 'active' | 'waiting' }) {
  if (state === 'done')    return <span className="step-icon done">✓</span>
  if (state === 'active')  return (
    <span className="step-icon active">
      <div style={{ width: 8, height: 8, border: '1.5px solid #D4908A', borderTopColor: '#0F0D0C', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
    </span>
  )
  return <span className="step-icon waiting">·</span>
}

function ProgressTracker({ stage }: { stage: ProgressStage }) {
  const currentIdx = STAGE_TO_STEP_INDEX[stage] ?? -1

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', height: '100%', gap: 0, padding: '40px 20px',
    }}>
      {/* Logo + heading */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16, marginBottom: 40 }}>
        <Logo size={56} radius={14} />
        <div style={{ textAlign: 'center' }}>
          <p className="display" style={{ fontSize: 22, fontWeight: 400, color: '#0F0D0C', margin: '0 0 6px' }}>
            Researching datasets…
          </p>
          <div className="dot-loader" style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
            <span /><span /><span />
          </div>
        </div>
      </div>

      {/* Step list */}
      <div style={{
        background: '#FAFAFA', border: '1px solid #EAE4E2', borderRadius: 14,
        padding: '18px 24px', minWidth: 300, maxWidth: 380,
      }}>
        {PIPELINE_STEPS.map((step, i) => {
          const state = i < currentIdx ? 'done' : i === currentIdx ? 'active' : 'waiting'
          return (
            <div
              key={step.id}
              className={`progress-step ${state}`}
              style={{
                animationDelay: `${i * 0.05}s`,
                opacity: state === 'waiting' ? 0.45 : 1,
              }}
            >
              <StepIcon state={state} />
              <div>
                <span style={{ fontWeight: state === 'active' ? 500 : 300 }}>{step.label}</span>
                {state === 'done' && step.sublabel && (
                  <span style={{ fontSize: 11, color: '#A89F9B', marginLeft: 6, fontWeight: 300 }}>
                    {step.sublabel}
                  </span>
                )}
              </div>
              {state === 'active' && (
                <span style={{ fontSize: 11, color: '#A89F9B', marginLeft: 'auto', fontFamily: "'Jost',sans-serif", fontWeight: 300 }}>
                  {i + 1}/{PIPELINE_STEPS.length}
                </span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ── SHARED DEEP PANEL HELPERS ────────────────────────────────── */
const sectionLabel = (text: string, color = '#6B5E58') => (
  <p style={{ fontFamily:"'Jost',sans-serif", fontSize:10, fontWeight:600, color, textTransform:'uppercase' as const, letterSpacing:'0.08em', margin:'0 0 6px' }}>{text}</p>
)
const divider = () => <div style={{ height:1, background:'#EAE4E2', margin:'12px 0' }} />

/* ── IMAGE DEEP QUALITY PANEL ─────────────────────────────────── */
function ImageDeepQuality({ da }: { da: DeepAnalysis }) {
  const dist    = da.class_distribution
  const pcs     = da.per_class_stats
  const total   = da.total_images

  const estImages  = da.total_images_est
  const estClasses = da.num_classes_est

  const classRows: { name: string; images: number; duplicates?: number; blurry?: number; corrupted?: number }[] =
    pcs
      ? pcs.map(s => ({ name: s.name, images: s.images, duplicates: s.duplicates, blurry: s.blurry, corrupted: s.corrupted }))
      : dist
        ? Object.entries(dist).sort((a,b) => b[1]-a[1]).map(([name,images]) => ({ name, images }))
        : []

  const maxImages   = classRows.length > 0 ? Math.max(...classRows.map(r => r.images)) : 1
  const totalImages = total ?? classRows.reduce((s,r) => s + r.images, 0)
  const minClass    = classRows.length > 0 ? classRows.reduce((a,b) => b.images < a.images ? b : a, classRows[0]) : null
  const maxClass    = classRows.length > 0 ? classRows[0] : null

  const totalDups    = da.duplicate_count ?? (pcs ? pcs.reduce((s,r) => s+r.duplicates, 0) : null)
  const totalBlurry  = da.blurry_count ?? (pcs ? pcs.reduce((s,r) => s+r.blurry, 0) : null)
  const totalCorrupt = da.corrupted_count ?? null

  const hasPCS = pcs && pcs.length > 0

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:0 }}>
      {(estImages != null || estClasses != null) && classRows.length === 0 && (
        <div style={{ marginBottom:8 }}>
          {sectionLabel('Dataset Scale (from description)')}
          <div style={{ display:'flex', gap:16, flexWrap:'wrap' as const, alignItems:'baseline', marginBottom:6 }}>
            {estImages != null && (
              <span style={{ fontFamily:"'Jost',sans-serif", fontSize:13 }}>
                <strong style={{ fontWeight:700, color:'#0F0D0C' }}>~{estImages.toLocaleString()}</strong>
                <span style={{ color:'#A89F9B', fontWeight:300, fontSize:12 }}> images</span>
              </span>
            )}
            {estClasses != null && (
              <span style={{ fontFamily:"'Jost',sans-serif", fontSize:13 }}>
                <strong style={{ fontWeight:700, color:'#0F0D0C' }}>{estClasses}</strong>
                <span style={{ color:'#A89F9B', fontWeight:300, fontSize:12 }}> classes</span>
              </span>
            )}
          </div>
          <div style={{ padding:'7px 10px', background:'#FFFAF7', border:'1px solid #F0E4D4', borderRadius:7 }}>
            <p style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#B8893A', margin:0, lineHeight:1.5 }}>
              ⓘ Class breakdown requires Kaggle API credentials to be configured.
            </p>
          </div>
        </div>
      )}
      {classRows.length > 0 && (
        <div>
          {sectionLabel('Class Balance')}
          <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
            {classRows.slice(0,10).map(row => {
              const pct   = totalImages > 0 ? (row.images / totalImages) * 100 : 0
              const isLow = pct < 10 && classRows.length > 2
              return (
                <div key={row.name} style={{ display:'grid', gridTemplateColumns: hasPCS ? '1fr 80px 64px 64px 64px' : '1fr 80px', gap:6, alignItems:'center', padding:'3px 0', borderBottom:'1px solid #F8F5F3' }}>
                  <div>
                    <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                      <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color: isLow ? '#B8893A' : '#2C2420', fontWeight: isLow ? 500 : 400, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{row.name}</span>
                      {isLow && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:9, color:'#B8893A', fontWeight:600, flexShrink:0 }}>LOW</span>}
                    </div>
                    <div style={{ height:3, background:'#EAE4E2', borderRadius:2, marginTop:2 }}>
                      <div style={{ width:`${(row.images/maxImages)*100}%`, height:'100%', background: isLow ? '#D4935A' : '#5A8A6A', borderRadius:2 }} />
                    </div>
                  </div>
                  <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, fontWeight:600, color:'#0F0D0C', textAlign:'right' as const, fontVariantNumeric:'tabular-nums' }}>{row.images.toLocaleString()}</span>
                  {hasPCS && (
                    <>
                      <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, fontWeight: row.duplicates! > 0 ? 600 : 300, color: row.duplicates! > 0 ? '#B8893A' : '#C8C2BE', textAlign:'right' as const }}>{row.duplicates?.toLocaleString() ?? '—'}</span>
                      <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, fontWeight: row.blurry! > 0 ? 600 : 300, color: row.blurry! > 0 ? '#7B8FA6' : '#C8C2BE', textAlign:'right' as const }}>{row.blurry?.toLocaleString() ?? '—'}</span>
                      <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, fontWeight: row.corrupted! > 0 ? 600 : 300, color: row.corrupted! > 0 ? '#C4615A' : '#C8C2BE', textAlign:'right' as const }}>{row.corrupted?.toLocaleString() ?? '—'}</span>
                    </>
                  )}
                </div>
              )
            })}
            {classRows.length > 10 && (
              <p style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#A89F9B', margin:'3px 0 0', fontStyle:'italic' }}>+ {classRows.length - 10} more classes</p>
            )}
          </div>
          {hasPCS && (
            <div style={{ display:'grid', gridTemplateColumns:'1fr 80px 64px 64px 64px', gap:6, marginTop:4 }}>
              <span />
              <span style={{ fontFamily:"'Jost',sans-serif", fontSize:9, fontWeight:600, color:'#A89F9B', textAlign:'right' as const, letterSpacing:'0.05em' }}>IMAGES</span>
              <span style={{ fontFamily:"'Jost',sans-serif", fontSize:9, fontWeight:600, color:'#B8893A', textAlign:'right' as const, letterSpacing:'0.05em' }}>DUPES</span>
              <span style={{ fontFamily:"'Jost',sans-serif", fontSize:9, fontWeight:600, color:'#7B8FA6', textAlign:'right' as const, letterSpacing:'0.05em' }}>BLURRY</span>
              <span style={{ fontFamily:"'Jost',sans-serif", fontSize:9, fontWeight:600, color:'#C4615A', textAlign:'right' as const, letterSpacing:'0.05em' }}>CORRUPT</span>
            </div>
          )}
          <div style={{ display:'grid', gridTemplateColumns: hasPCS ? '1fr 80px 64px 64px 64px' : '1fr 80px', gap:6, paddingTop:6, marginTop:4, borderTop:'1.5px solid #D8CFC9' }}>
            <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, fontWeight:600, color:'#6B5E58' }}>Total · {classRows.length} classes</span>
            <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, fontWeight:600, color:'#0F0D0C', textAlign:'right' as const, fontVariantNumeric:'tabular-nums' }}>{totalImages.toLocaleString()}</span>
            {hasPCS && totalDups !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, fontWeight:600, color: totalDups > 0 ? '#B8893A' : '#A89F9B', textAlign:'right' as const }}>{totalDups.toLocaleString()}</span>)}
            {hasPCS && totalBlurry !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, fontWeight:600, color: totalBlurry > 0 ? '#7B8FA6' : '#A89F9B', textAlign:'right' as const }}>{totalBlurry.toLocaleString()}</span>)}
            {hasPCS && totalCorrupt !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, fontWeight:600, color: totalCorrupt > 0 ? '#C4615A' : '#A89F9B', textAlign:'right' as const }}>{totalCorrupt.toLocaleString()}</span>)}
          </div>
        </div>
      )}
      {divider()}
      {(totalDups !== null || da.duplicate_pct !== null) && (
        <div>
          {sectionLabel('Duplicate Images', totalDups && totalDups > 0 ? '#B8893A' : '#6B5E58')}
          <div style={{ display:'flex', gap:16, flexWrap:'wrap' as const, alignItems:'baseline' }}>
            {totalDups !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:13 }}><strong style={{ fontWeight:700, color: totalDups > 0 ? '#B8893A' : '#0F0D0C' }}>{totalDups.toLocaleString()}</strong><span style={{ color:'#A89F9B', fontWeight:300, fontSize:12 }}> duplicate images</span></span>)}
            {da.duplicate_pct !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#A89F9B', fontWeight:300 }}>({da.duplicate_pct.toFixed(1)}% of dataset)</span>)}
          </div>
        </div>
      )}
      {(da.blur_pct !== null || totalBlurry !== null) && (
        <>
          {divider()}
          <div>
            {sectionLabel('Blur Quality', da.blur_pct && da.blur_pct >= 10 ? '#7B8FA6' : '#6B5E58')}
            <div style={{ display:'flex', gap:16, flexWrap:'wrap' as const, alignItems:'baseline' }}>
              {da.blur_pct !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:13 }}><strong style={{ fontWeight:700, color: da.blur_pct >= 10 ? '#7B8FA6' : '#0F0D0C' }}>{da.blur_pct.toFixed(1)}%</strong><span style={{ color:'#A89F9B', fontWeight:300, fontSize:12 }}> blurry</span></span>)}
              {totalBlurry !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#A89F9B', fontWeight:300 }}>({totalBlurry.toLocaleString()} images in sample)</span>)}
            </div>
          </div>
        </>
      )}
      {(da.corrupted_pct !== null || da.corrupted_count !== null) && (
        <>
          {divider()}
          <div>
            {sectionLabel('Corrupted Images', da.corrupted_count && da.corrupted_count > 0 ? '#C4615A' : '#6B5E58')}
            <div style={{ display:'flex', gap:16, flexWrap:'wrap' as const, alignItems:'baseline' }}>
              {da.corrupted_count !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:13 }}><strong style={{ fontWeight:700, color: da.corrupted_count > 0 ? '#C4615A' : '#0F0D0C' }}>{da.corrupted_count.toLocaleString()}</strong><span style={{ color:'#A89F9B', fontWeight:300, fontSize:12 }}> corrupted</span></span>)}
              {da.corrupted_pct !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#A89F9B', fontWeight:300 }}>({da.corrupted_pct.toFixed(1)}% of sample)</span>)}
            </div>
          </div>
        </>
      )}
      {da.dimension_info && (
        <>
          {divider()}
          <div>
            {sectionLabel('Resolution Distribution')}
            <div style={{ display:'flex', gap:16, flexWrap:'wrap' as const }}>
              {da.dimension_info.dominant_size && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#2C2420', fontWeight:400 }}><strong style={{ fontWeight:600 }}>{da.dimension_info.dominant_size[0]}×{da.dimension_info.dominant_size[1]}</strong><span style={{ color:'#A89F9B', fontWeight:300 }}> dominant</span></span>)}
              <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#2C2420', fontWeight:400 }}><strong style={{ fontWeight:600 }}>{da.dimension_info.unique_sizes}</strong><span style={{ color:'#A89F9B', fontWeight:300 }}> unique sizes</span></span>
              {da.dimension_info.undersized_count > 0 && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#C4615A', fontWeight:500 }}>⚠ {da.dimension_info.undersized_count} undersized (&lt;32px)</span>)}
              {da.dimension_info.is_consistent === true && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#7A9E8A' }}>✓ Consistent sizes</span>)}
              {da.dimension_info.is_consistent === false && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#B8893A' }}>⚠ Mixed sizes detected</span>)}
            </div>
          </div>
        </>
      )}
      {da.geographic_bias?.detected && (
        <>
          {divider()}
          <div>
            {sectionLabel('Geographic Bias', '#B8893A')}
            <p style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#B8893A', margin:0, fontWeight:400 }}>
              🌍 {da.geographic_bias.warning ?? `Dataset may be regionally biased (${da.geographic_bias.regions.join(', ')})`}
            </p>
          </div>
        </>
      )}
      {minClass && maxClass && maxClass.images > 0 && (minClass.images / maxClass.images) < 0.5 && (
        <>
          {divider()}
          <div style={{ padding:'8px 12px', background:'#FFFAF7', border:'1px solid #F0E4D4', borderRadius:8 }}>
            <p style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#B8893A', margin:0, fontWeight:400 }}>
              ⚠ Class imbalance — <strong>{minClass.name}</strong> has only {minClass.images.toLocaleString()} images vs <strong>{maxClass.name}</strong> with {maxClass.images.toLocaleString()}.
            </p>
          </div>
        </>
      )}
      {da.is_partial && (
        <>
          {divider()}
          <p style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#A89F9B', margin:0, fontStyle:'italic' }}>ⓘ Partial analysis · Full details require dataset download</p>
        </>
      )}
    </div>
  )
}

/* ── TABULAR DEEP QUALITY PANEL ───────────────────────────────── */
function TabularDeepQuality({ da }: { da: DeepAnalysis }) {
  const dist     = da.class_distribution
  const nullCols = da.null_columns ?? []
  const detail   = da.columns_detail ?? []
  const cols     = da.columns

  const lowInfoCols   = detail.filter(c => c.is_id || c.cardinality_ratio > 0.95)
  const mixedTypeCols = detail.filter(c => c.type === 'mixed')
  const constantCols  = detail.filter(c => c.unique_count <= 1 && !c.is_target)

  const dupCount = da.duplicate_count
  const dupPct   = da.duplicate_pct

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:0 }}>
      {nullCols.length > 0 ? (
        <div>
          {sectionLabel('Missing Values', '#C4615A')}
          <div style={{ display:'flex', flexDirection:'column', gap:3 }}>
            {nullCols.map(col => (
              <div key={col.name} style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'3px 0', borderBottom:'1px solid #F8F5F3' }}>
                <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#2C2420', fontWeight:400 }}>{col.name}</span>
                <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                  {col.null_count != null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, fontWeight:600, color:'#C4615A', fontVariantNumeric:'tabular-nums' }}>{col.null_count.toLocaleString()} nulls</span>)}
                  <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#A89F9B', fontWeight:300, width:40, textAlign:'right' as const }}>{col.missing_pct.toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div>
          {sectionLabel('Missing Values')}
          <p style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#7A9E8A', margin:0 }}>✓ No significant missing values</p>
        </div>
      )}
      {divider()}
      <div>
        {sectionLabel('Duplicate Rows', dupCount && dupCount > 0 ? '#B8893A' : '#6B5E58')}
        {dupCount !== null || dupPct !== null ? (
          <div style={{ display:'flex', gap:14, alignItems:'baseline', flexWrap:'wrap' as const }}>
            {dupCount !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:13 }}><strong style={{ fontWeight:700, color: dupCount > 0 ? '#B8893A' : '#0F0D0C' }}>{dupCount.toLocaleString()}</strong><span style={{ color:'#A89F9B', fontWeight:300, fontSize:12 }}> duplicate rows</span></span>)}
            {dupPct !== null && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#A89F9B', fontWeight:300 }}>({dupPct.toFixed(1)}%)</span>)}
            {(dupCount ?? 0) === 0 && (<span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#7A9E8A' }}>✓ No duplicates</span>)}
          </div>
        ) : (<p style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#A89F9B', margin:0, fontStyle:'italic' }}>Not analysed</p>)}
      </div>
      {dist && Object.keys(dist).length > 0 && (
        <>
          {divider()}
          <div>
            {sectionLabel('Class Balance')}
            <div style={{ display:'flex', flexWrap:'wrap' as const, gap:'6px 20px' }}>
              {Object.entries(dist).sort((a,b) => b[1]-a[1]).map(([cls, count]) => (
                <span key={cls} style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#2C2420' }}>
                  <span style={{ fontWeight:300, color:'#6B5E58' }}>{cls}: </span>
                  <strong style={{ fontWeight:600 }}>{count.toLocaleString()}</strong>
                </span>
              ))}
            </div>
          </div>
        </>
      )}
      {cols && cols.total !== null && (
        <>
          {divider()}
          <div>
            {sectionLabel('Column Types')}
            <div style={{ display:'flex', flexWrap:'wrap' as const, gap:'4px 18px' }}>
              {[
                { label:'Total',       val: cols.total,       color:'#0F0D0C' },
                { label:'Numeric',     val: cols.numeric,     color:'#7B8FA6' },
                { label:'Categorical', val: cols.categorical, color:'#7A9E8A' },
                { label:'Text',        val: cols.text,        color:'#C8A882' },
              ].filter(d => d.val !== null && d.val! > 0).map(({ label, val, color }) => (
                <span key={label} style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#6B5E58', fontWeight:300 }}>
                  <strong style={{ fontWeight:600, color }}>{val}</strong> {label.toLowerCase()}
                </span>
              ))}
            </div>
          </div>
        </>
      )}
      {mixedTypeCols.length > 0 && (
        <>
          {divider()}
          <div>
            {sectionLabel('Type Inconsistencies', '#B8893A')}
            <div style={{ display:'flex', flexWrap:'wrap' as const, gap:6 }}>
              {mixedTypeCols.map(col => (
                <span key={col.name} style={{ fontFamily:"'Jost',sans-serif", fontSize:11, padding:'3px 10px', background:'#FFFAF7', border:'1px solid #F0E4D4', borderRadius:4, color:'#B8893A' }}>
                  ⚠ {col.name} <span style={{ fontWeight:300, color:'#C8A882' }}>(mixed types)</span>
                </span>
              ))}
            </div>
          </div>
        </>
      )}
      {da.geographic_bias?.detected && (
        <>
          {divider()}
          <div>
            {sectionLabel('Geographic Bias', '#B8893A')}
            <p style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#B8893A', margin:0 }}>
              🌍 {da.geographic_bias.warning ?? `Possible single-region bias: ${da.geographic_bias.regions.join(', ')}`}
            </p>
          </div>
        </>
      )}
    </div>
  )
}

/* ── TRAINING RISKS / SUGGESTED FIXES ────────────────────────── */
function SuggestedFixes({ fixes }: { fixes: string[] }) {
  if (!fixes.length) return null
  return (
    <div>
      {sectionLabel('Training Risks & Suggested Fixes', '#2C2420')}
      <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
        {fixes.map((fix, i) => (
          <div key={i} style={{ display:'flex', gap:8, padding:'6px 10px', background:'#FFFAF7', border:'1px solid #F0E4D4', borderRadius:7 }}>
            <span style={{ color:'#B8893A', flexShrink:0, fontSize:12 }}>→</span>
            <span style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:12, color:'#2C2420', lineHeight:1.55 }}>{fix}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── SCORE BAR ────────────────────────────────────────────────── */
function ScoreBar({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  const pct = Math.round(value * 100)
  const fillColor = pct >= 70 ? '#7A9E8A' : pct >= 40 ? '#C8A882' : '#D4908A'
  return (
    <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:6 }}>
      <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, fontWeight:highlight?500:300, color:highlight?'#2C2420':'#6B5E58', width:140, flexShrink:0 }}>{label}</span>
      <div className="score-track">
        <div className="score-fill" style={{ width:`${pct}%`, background:fillColor }} />
      </div>
      <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, fontWeight:highlight?500:300, color:highlight?'#2C2420':'#A89F9B', width:32, textAlign:'right' as const }}>{pct}%</span>
    </div>
  )
}

/* ── RISK BADGE ───────────────────────────────────────────────── */
function RiskBadge({ risk }: { risk: string }) {
  const map: Record<string, { text: string; color: string; bg: string; border: string }> = {
    missing_description: { text:'No description', color:'#C4615A', bg:'#FDF4F4', border:'#F0D0CE' },
    no_license:          { text:'No license',     color:'#B8893A', bg:'#FDF9F2', border:'#EDD9B0' },
    potentially_stale:   { text:'Stale data',     color:'#B8893A', bg:'#FDF9F2', border:'#EDD9B0' },
    low_engagement:      { text:'Low engagement', color:'#A89F9B', bg:'#F8F4F0', border:'#EAE4E2' },
  }
  const m = map[risk] || { text:risk, color:'#A89F9B', bg:'#F8F4F0', border:'#EAE4E2' }
  return <span className="tag" style={{ background:m.bg, color:m.color, border:`1px solid ${m.border}` }}>⚠ {m.text}</span>
}

/* ── DATASET CARD ─────────────────────────────────────────────── */
function DatasetCard({ ds, insight, idx }: { ds: DatasetResult; insight?: any; idx: number }) {
  const [open, setOpen] = useState(false)
  const score = Math.round(ds.composite_score * 100)
  const scoreStyle = score >= 70 ? { color:'#3D7A5C', bg:'#F0F7F3', border:'#C8DDD2' }
    : score >= 50 ? { color:'#B8893A', bg:'#FFFAF7', border:'#F0E4D4' }
    : { color:'#C4615A', bg:'#FDF4F4', border:'#F0D0CE' }
  const src = SOURCE_META[ds.source] || { label:ds.source, color:'#A89F9B', bg:'#F8F4F0', border:'#EAE4E2' }
  const bd  = ds.score_breakdown
  const dimScores = [
    { key:'task_relevance',    label:'Task relevance', value:bd.task_relevance },
    { key:'quality',           label:'Data quality',   value:bd.quality },
    { key:'popularity',        label:'Popularity',     value:bd.popularity },
    { key:'freshness',         label:'Freshness',      value:bd.freshness },
    { key:'description_match', label:'Query match',    value:bd.description_match },
  ]
  const topDim    = [...dimScores].sort((a,b) => b.value-a.value)[0]
  const bottomDim = [...dimScores].sort((a,b) => a.value-b.value)[0]
  const da = ds.deep_analysis

  const isImage   = !!(da?.class_distribution || da?.per_class_stats || da?.total_images != null || da?.total_images_est != null || da?.num_classes_est != null)
  const isTabular = !isImage && !!((da?.null_columns?.length ?? 0) > 0 || (da?.columns_detail?.length ?? 0) > 0 || da?.duplicate_pct != null || da?.duplicate_count != null)

  const mutedLink: React.CSSProperties = { fontSize:11, color:'#A89F9B', textDecoration:'none', fontFamily:"'Jost',sans-serif", fontWeight:400, transition:'color 0.12s' }

  return (
    <div className="dataset-card fade-up" style={{ animationDelay:`${idx*0.04}s` }}>
      <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between', gap:14, marginBottom:10 }}>
        <div style={{ display:'flex', alignItems:'center', gap:10, minWidth:0, flex:1 }}>
          <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, fontWeight:500, color:'#A89F9B', flexShrink:0, fontVariantNumeric:'tabular-nums' }}>{String(idx+1).padStart(2,'0')}</span>
          <span className="display" style={{ fontSize:15, fontWeight:600, color:'#0F0D0C', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', letterSpacing:'-0.2px' }}>{ds.title}</span>
          <span className="tag" style={{ background:src.bg, color:src.color, border:`1px solid ${src.border}` }}>{src.label}</span>
        </div>
        <div style={{ background:scoreStyle.bg, border:`1px solid ${scoreStyle.border}`, borderRadius:10, padding:'4px 12px', flexShrink:0, textAlign:'center' }}>
          <span className="display" style={{ fontSize:22, fontWeight:600, color:scoreStyle.color, lineHeight:1 }}>{score}</span>
          <div style={{ fontFamily:"'Jost',sans-serif", fontSize:9, color:'#A89F9B', marginTop:1, letterSpacing:'0.04em' }}>/100</div>
        </div>
      </div>
      {ds.risk_indicators?.length > 0 && (
        <div style={{ display:'flex', flexWrap:'wrap', gap:5, marginBottom:8 }}>
          {ds.risk_indicators.map(r => <RiskBadge key={r} risk={r} />)}
        </div>
      )}
      {da && (
        <div style={{ display:'flex', flexWrap:'wrap', gap:5, marginBottom: 8 }}>
          {(da.blur_pct ?? 0) > 10 && <span className="tag" style={{ background:'#FDF4F4', color:'#C4615A', border:'1px solid #F0D0CE' }}>⚠ {da.blur_pct!.toFixed(0)}% blurry</span>}
          {(da.duplicate_pct ?? 0) > 1 && <span className="tag" style={{ background:'#FFFAF7', color:'#B8893A', border:'1px solid #F0E4D4' }}>⚠ {da.duplicate_pct!.toFixed(1)}% dupes</span>}
          {da.geographic_bias?.detected && <span className="tag" style={{ background:'#F8F4F0', color:'#6B5E58', border:'1px solid #EAE4E2' }}>⚠ Region: {da.geographic_bias.regions[0]}</span>}
          {da.total_images != null && <span className="tag" style={{ background:'#F2F7F4', color:'#3D7A5C', border:'1px solid #C8DDD2' }}>{da.total_images.toLocaleString()} images</span>}
          {da.num_classes != null && <span className="tag" style={{ background:'#F4F5F8', color:'#7B8FA6', border:'1px solid #CDD2DC' }}>{da.num_classes} classes</span>}
        </div>
      )}
      {ds.description && <p style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, color:'#6B5E58', fontSize:13, margin:'0 0 12px', lineHeight:1.65, display:'-webkit-box', WebkitLineClamp:2, WebkitBoxOrient:'vertical', overflow:'hidden' }}>{ds.description}</p>}
      {insight?.why_ranked && (
        <div style={{ background:'#F8F4F0', borderLeft:'2px solid #D4908A', borderRadius:'0 8px 8px 0', padding:'8px 14px', marginBottom:12 }}>
          <span style={{ fontFamily:"'Jost',sans-serif", fontSize:10, fontWeight:600, color:'#0F0D0C', letterSpacing:'0.07em', textTransform:'uppercase' }}>✦ Why ranked here  </span>
          <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, fontWeight:300, color:'#2C2420', lineHeight:1.6 }}>{insight.why_ranked}</span>
        </div>
      )}
      <div style={{ display:'flex', flexWrap:'wrap', gap:16, marginBottom:10, alignItems:'center' }}>
        {ds.row_count    && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#A89F9B', fontWeight:300 }}><span style={{ color:'#2C2420', fontWeight:500 }}>{ds.row_count.toLocaleString()}</span> rows</span>}
        {ds.column_count && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#A89F9B', fontWeight:300 }}><span style={{ color:'#2C2420', fontWeight:500 }}>{ds.column_count}</span> features</span>}
        {ds.download_count && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#A89F9B', fontWeight:300 }}><span style={{ color:'#2C2420', fontWeight:500 }}>{ds.download_count.toLocaleString()}</span> downloads</span>}
        {ds.last_updated && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#A89F9B', fontWeight:300 }}>Updated <span style={{ color:'#2C2420', fontWeight:500 }}>{new Date(ds.last_updated).getFullYear()}</span></span>}
        {ds.license && <span className="tag" style={{ background:'#F8F4F0', color:'#6B5E58', border:'1px solid #EAE4E2', fontSize:10 }}>{ds.license}</span>}
      </div>
      {ds.tags?.length > 0 && <div style={{ display:'flex', flexWrap:'wrap', gap:5, marginBottom:10 }}>{ds.tags.slice(0,6).map(t => <span key={t} className="tag" style={{ background:'#F8F4F0', color:'#6B5E58', border:'1px solid #EAE4E2', fontSize:10 }}>{t}</span>)}</div>}
      <div style={{ display:'flex', gap:6, marginBottom:12, flexWrap:'wrap', alignItems:'center' }}>
        <span className="tag" style={{ background:'#F0F7F3', color:'#3D7A5C', border:'1px solid #C8DDD2' }}>↑ {topDim.label} {Math.round(topDim.value*100)}%</span>
        <span className="tag" style={{ background:'#FFFAF7', color:'#B8893A', border:'1px solid #F0E4D4' }}>↓ {bottomDim.label} {Math.round(bottomDim.value*100)}%</span>
        <div style={{ flex:1 }} />
        {ds.has_description && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:10, color:'#7A9E8A' }}>✓ Description</span>}
        {ds.has_license_info && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:10, color:'#7A9E8A' }}>✓ License</span>}
        {!ds.has_description && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:10, color:'#D4908A' }}>✗ No description</span>}
      </div>
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', borderTop:'1px solid #EAE4E2', paddingTop:12 }}>
        <button onClick={() => setOpen(!open)} style={{ background:'none', border:'none', ...mutedLink, cursor:'pointer', padding:0 }}
          onMouseEnter={e => (e.currentTarget.style.color='#6B5E58')} onMouseLeave={e => (e.currentTarget.style.color='#A89F9B')}>
          {open ? '▲ Hide analysis' : '▼ Full score analysis'}
        </button>
        <a href={ds.source_url} target="_blank" rel="noopener noreferrer" style={mutedLink}
          onMouseEnter={e => (e.currentTarget.style.color='#6B5E58')} onMouseLeave={e => (e.currentTarget.style.color='#A89F9B')}>
          View on {src.label} →
        </a>
      </div>
      {open && (
        <div style={{ marginTop:16, paddingTop:16, borderTop:'1px solid #EAE4E2' }}>
          <p className="section-label">Score Breakdown</p>
          {dimScores.map(d => <ScoreBar key={d.key} label={d.label} value={d.value} highlight={d.key===topDim.key} />)}
          {ds.freshness_explanation && (
            <div style={{ marginTop:14 }}>
              <p className="section-label">Freshness</p>
              <p style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:12, color:'#6B5E58', lineHeight:1.6, background:'#F8F4F0', borderRadius:8, padding:'8px 12px', margin:0 }}>{ds.freshness_explanation}</p>
            </div>
          )}
          {da && (isImage || isTabular || da.suggested_fixes.length > 0) && (
            <div style={{ marginTop:16 }}>
              <p className="section-label">Deep Quality Analysis</p>
              <div style={{ background:'#F8F4F0', border:'1px solid #EAE4E2', borderRadius:10, padding:'14px 16px', display:'flex', flexDirection:'column', gap:0 }}>
                {isImage   && <ImageDeepQuality   da={da} />}
                {isTabular && <TabularDeepQuality da={da} />}
                {da.suggested_fixes.length > 0 && (
                  <>
                    {(isImage || isTabular) && <div style={{ height:1, background:'#EAE4E2', margin:'12px 0' }} />}
                    <SuggestedFixes fixes={da.suggested_fixes} />
                  </>
                )}
                {!isImage && !isTabular && (
                  <div style={{ display:'flex', flexWrap:'wrap', gap:16 }}>
                    {da.completeness_score !== null && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#6B5E58', fontWeight:300 }}>Completeness: <strong style={{ fontWeight:600, color:'#0F0D0C' }}>{Math.round(da.completeness_score)}%</strong></span>}
                    {da.balance_score      !== null && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#6B5E58', fontWeight:300 }}>Balance: <strong style={{ fontWeight:600, color: da.balance_score < 60 ? '#C4615A' : '#0F0D0C' }}>{Math.round(da.balance_score)}%</strong></span>}
                    {da.uniqueness_score   !== null && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#6B5E58', fontWeight:300 }}>Uniqueness: <strong style={{ fontWeight:600, color:'#0F0D0C' }}>{Math.round(da.uniqueness_score)}%</strong></span>}
                  </div>
                )}
              </div>
            </div>
          )}
          {insight?.why_ranked && (
            <div style={{ marginTop:14 }}>
              <p className="section-label">Why Ranked Here</p>
              <div style={{ background:'#F8F4F0', borderLeft:'2px solid #D4908A', borderRadius:'0 8px 8px 0', padding:'8px 14px' }}>
                <p style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:12, color:'#2C2420', margin:0, lineHeight:1.65 }}>{insight.why_ranked}</p>
              </div>
            </div>
          )}
          {insight?.strengths?.length > 0 && (
            <div style={{ marginTop:12 }}>
              <p className="section-label">Strengths</p>
              {insight.strengths.slice(0,3).map((s: string, i: number) => (
                <p key={i} style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:12, color:'#7A9E8A', margin:'3px 0', lineHeight:1.55 }}>✓ {s}</p>
              ))}
            </div>
          )}
          {insight?.weaknesses?.length > 0 && (
            <div style={{ marginTop:10 }}>
              <p className="section-label">Limitations</p>
              {insight.weaknesses.slice(0,2).map((w: string, i: number) => (
                <p key={i} style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:12, color:'#B8893A', margin:'3px 0', lineHeight:1.55 }}>⚠ {w}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ── PROGRESS SIMULATION HOOK ─────────────────────────────────── */
// Simulates realistic stage timing while the real fetch is in-flight.
// If backend ever emits SSE progress events, replace this with a real listener.
function useProgressSimulation(loading: boolean): ProgressStage {
  const [stage, setStage] = useState<ProgressStage>('IDLE')

  useEffect(() => {
    if (!loading) { setStage('IDLE'); return }

    // Staggered timing (ms) that matches expected real backend duration
    const TIMELINE: Array<[number, ProgressStage]> = [
      [0,    'SEARCHING'],
      [800,  'KAGGLE_DONE'],
      [1400, 'HF_DONE'],
      [2000, 'OPENML_DONE'],
      [2400, 'DEDUPLICATING'],
      [3000, 'METADATA_RANKING'],
      [3800, 'LIGHT_ANALYSIS'],
      [5000, 'RERANKING'],
      [6000, 'DEEP_ANALYSIS'],
      [8000, 'INSIGHTS'],
    ]

    setStage('SEARCHING')
    const timers = TIMELINE.slice(1).map(([delay, s]) =>
      setTimeout(() => setStage(s), delay)
    )
    return () => timers.forEach(clearTimeout)
  }, [loading])

  return stage
}

/* ── MAIN APP ─────────────────────────────────────────────────── */
export default function Home() {
  const [query, setQuery]                 = useState('')
  const [loading, setLoading]             = useState(false)
  const [result, setResult]               = useState<SearchResponse | null>(null)
  const [error, setError]                 = useState<string | null>(null)

  // Session history — each entry is one search in the current conversation
  const [sessions, setSessions]           = useState<SearchSession[]>([])

  // The active sidebar item (purely visual — never drives a new chat)
  const [activeSession, setActiveSession] = useState<string | null>(null)

  // Tracks the last query for conversation continuity (backend context)
  const [lastQuery, setLastQuery]         = useState<string | null>(null)

  // Full chat history in this conversation — never reset unless New Search clicked
  const [chatHistory, setChatHistory]     = useState<Array<{ query: string; result: SearchResponse }>>([])

  const inputRef = useRef<HTMLInputElement>(null)

  // Progress animation
  const progressStage = useProgressSimulation(loading)

  // ── doSearch: stays in the SAME conversation, never creates new chat
  const doSearch = async (q: string) => {
    const trimmed = q.trim()
    if (!trimmed || loading) return
    setQuery(trimmed)
    setLoading(true)
    setError(null)
    // Do NOT reset result here — keep current results visible until new ones arrive
    try {
      const data = await searchDatasets(trimmed, 10, ['huggingface','openml','kaggle'], lastQuery || undefined)
      setResult(data)
      setLastQuery(trimmed)
      setChatHistory(prev => [...prev, { query: trimmed, result: data }])
      const session: SearchSession = {
        id:          Date.now().toString(),
        query:       trimmed,
        timestamp:   new Date(),
        resultCount: data.returned || 0,
      }
      setSessions(prev => [session, ...prev.slice(0, 19)])
      setActiveSession(session.id)
    } catch (e: any) {
      const msg = e?.message || ''
      setError(
        msg.includes('fetch') || msg.includes('NetworkError') || msg.includes('Failed to fetch')
          ? 'Cannot reach the DataScout backend at localhost:8000. Make sure uvicorn is running.'
          : msg || 'Something went wrong. Check the browser console.'
      )
    } finally {
      setLoading(false)
    }
  }

  // ── startNewConversation: called ONLY by explicit "New Search" button
  const startNewConversation = () => {
    setResult(null)
    setError(null)
    setQuery('')
    setActiveSession(null)
    setLastQuery(null)
    setChatHistory([])
    setTimeout(() => inputRef.current?.focus(), 50)
  }

  const getInsight = (ds: DatasetResult) =>
    result?.intelligence?.dataset_insights?.find((i: any) => i.dataset_id === ds.dataset_id)

  return (
    <>
      <div style={{ display:'flex', height:'100vh', overflow:'hidden', fontFamily:"'Jost', sans-serif" }}>

        {/* ── SIDEBAR ── */}
        <div className="sidebar" style={{ width:252, display:'flex', flexDirection:'column', flexShrink:0 }}>
          <div style={{ padding:'20px 18px 16px', borderBottom:'1px solid #EAE4E2' }}>
            <div style={{ display:'flex', alignItems:'center', gap:11 }}>
              <Logo size={36} radius={10} />
              <div>
                <div className="display" style={{ fontSize:17, fontWeight:600, color:'#0F0D0C', letterSpacing:'-0.3px', lineHeight:1 }}>DataScout</div>
                <div style={{ fontFamily:"'Jost',sans-serif", fontSize:9, fontWeight:400, color:'#A89F9B', marginTop:3, letterSpacing:'0.09em', textTransform:'uppercase' }}>Dataset Intelligence</div>
              </div>
            </div>
          </div>
          <div style={{ padding:'10px 10px 4px' }}>
            {/* ─ NEW SEARCH: the ONLY place a new conversation is created ─ */}
            <button className="btn-ghost" onClick={startNewConversation}>
              <span style={{ fontSize:16, lineHeight:1, color:'#0F0D0C' }}>+</span> New search
            </button>
          </div>
          <div style={{ flex:1, overflowY:'auto', padding:'0 10px 8px' }}>
            {sessions.length > 0 && (
              <>
                <p className="section-label" style={{ padding:'10px 8px 4px' }}>This conversation</p>
                {sessions.map(s => (
                  <div key={s.id}
                    onClick={() => setActiveSession(s.id)}
                    style={{ padding:'9px 10px', borderRadius:9, cursor:'pointer', marginBottom:2, background:activeSession===s.id?'#F8F4F0':'transparent', border:`1px solid ${activeSession===s.id?'#EAE4E2':'transparent'}`, transition:'all 0.12s' }}
                    onMouseEnter={e => { if (activeSession!==s.id) e.currentTarget.style.background='#F8F4F0' }}
                    onMouseLeave={e => { if (activeSession!==s.id) e.currentTarget.style.background='transparent' }}>
                    <p style={{ margin:0, fontFamily:"'Jost',sans-serif", fontWeight:400, fontSize:12, color:'#0F0D0C', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{s.query}</p>
                    <p style={{ margin:'2px 0 0', fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:10, color:'#A89F9B' }}>{s.resultCount} results</p>
                  </div>
                ))}
              </>
            )}
          </div>
          <div style={{ padding:'14px 18px', borderTop:'1px solid #EAE4E2' }}>
            <p className="section-label">Data sources</p>
            {(['huggingface','kaggle','openml'] as const).map(src => (
              <div key={src} style={{ display:'flex', alignItems:'center', gap:7, marginBottom:5 }}>
                <div style={{ width:5, height:5, borderRadius:'50%', background:SOURCE_META[src].color, flexShrink:0 }} />
                <span style={{ fontFamily:"'Jost',sans-serif", fontWeight:400, fontSize:11, color:'#2C2420' }}>{SOURCE_META[src].label}</span>
              </div>
            ))}
            <div style={{ height:1, background:'#EAE4E2', margin:'10px 0 8px' }} />
            <p className="section-label">Powered by</p>
            <div style={{ display:'flex', alignItems:'center', gap:7, marginBottom:5 }}>
              <div style={{ width:5, height:5, borderRadius:'50%', background:'#4285F4', flexShrink:0 }} />
              <span style={{ fontFamily:"'Jost',sans-serif", fontWeight:400, fontSize:11, color:'#2C2420' }}>Gemini AI</span>
            </div>
            <div style={{ display:'flex', alignItems:'center', gap:7, marginBottom:8 }}>
              <div style={{ width:5, height:5, borderRadius:'50%', background:'#F0A500', flexShrink:0 }} />
              <span style={{ fontFamily:"'Jost',sans-serif", fontWeight:400, fontSize:11, color:'#2C2420' }}>Elasticsearch</span>
            </div>
            <p style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:9, color:'#A89F9B', marginTop:2, letterSpacing:'0.04em' }}>Deterministic ranking · Grounded AI</p>
          </div>
        </div>

        {/* ── MAIN CONTENT ── */}
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', background:'transparent' }}>
          <div style={{ flex:1, overflowY:'auto', padding:result?'28px 40px 24px':'0' }}>

            {/* Empty state */}
            {!result && !loading && !error && (
              <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:'100%', padding:48 }}>
                <div style={{ marginBottom:28 }}>
                  <Logo size={72} radius={18} />
                </div>
                <h1 className="display" style={{ fontSize:32, fontWeight:600, color:'#0F0D0C', margin:'0 0 10px', textAlign:'center', letterSpacing:'-0.5px', lineHeight:1.2 }}>What are you building?</h1>
                <p style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:14, color:'#6B5E58', margin:'0 0 36px', textAlign:'center', maxWidth:460, lineHeight:1.7 }}>
                  Describe your ML task and DataScout will discover the right datasets across Kaggle, HuggingFace, and OpenML.
                </p>
                <div style={{ display:'flex', flexDirection:'column', gap:6, width:'100%', maxWidth:500 }}>
                  <p className="section-label" style={{ textAlign:'center' }}>Try an example</p>
                  {EXAMPLES.map(ex => (
                    <button key={ex} onClick={() => doSearch(ex)}
                      style={{ background:'#FAFAFA', border:'1px solid #EAE4E2', borderRadius:10, padding:'11px 16px', fontFamily:"'Jost',sans-serif", fontWeight:300, color:'#6B5E58', fontSize:13, cursor:'pointer', textAlign:'left', transition:'all 0.15s' }}
                      onMouseEnter={e => { e.currentTarget.style.background='#F8F4F0'; e.currentTarget.style.borderColor='#D8CFC9'; e.currentTarget.style.color='#0F0D0C'; e.currentTarget.style.transform='translateX(4px)' }}
                      onMouseLeave={e => { e.currentTarget.style.background='#FAFAFA'; e.currentTarget.style.borderColor='#EAE4E2'; e.currentTarget.style.color='#6B5E58'; e.currentTarget.style.transform='translateX(0)' }}>
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* ── PROGRESS TRACKER (replaces old loading spinner + jargon text) ── */}
            {loading && <ProgressTracker stage={progressStage} />}

            {/* Error */}
            {error && !loading && (
              <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100%', padding:40 }}>
                <div style={{ background:'#FDF4F4', border:'1px solid #F0D0CE', borderRadius:12, padding:'16px 22px', maxWidth:500 }}>
                  <p style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, color:'#C4615A', margin:0, fontSize:13, lineHeight:1.65 }}>{error}</p>
                </div>
              </div>
            )}

            {/* Results */}
            {result && !loading && (
              <div style={{ display:'flex', gap:24 }}>
                <div style={{ flex:1, minWidth:0 }}>
                  <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:16, flexWrap:'wrap', gap:8 }}>
                    <div style={{ display:'flex', alignItems:'baseline', gap:8 }}>
                      <span className="display" style={{ fontSize:20, fontWeight:600, color:'#0F0D0C', letterSpacing:'-0.3px' }}>{result.returned} datasets</span>
                      <span style={{ fontFamily:"'Jost',sans-serif", fontSize:12, color:'#A89F9B', fontWeight:300 }}>from {result.total_found} candidates</span>
                    </div>
                    <div style={{ display:'flex', alignItems:'center', gap:14 }}>
                      {result.processing_time_ms && <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#A89F9B', fontWeight:300 }}>{result.processing_time_ms}ms</span>}
                      <span style={{ fontFamily:"'Jost',sans-serif", fontSize:11, color:'#A89F9B', fontWeight:300 }}>
                        Confidence <span style={{ fontWeight:500, color:result.confidence==='high'?'#3D7A5C':result.confidence==='medium'?'#B8893A':'#C4615A' }}>{result.confidence}</span>
                      </span>
                      {result.intelligence_available && <span className="tag" style={{ background:'#F0FDF4', color:'#15803D', border:'1px solid #BBF7D0', fontWeight:500 }}>✦ Gemini AI</span>}
                      {result.retrieval_method==='elasticsearch_hybrid' && <span className="tag" style={{ background:'#EFF6FF', color:'#1D4ED8', border:'1px solid #BFDBFE', fontWeight:500 }}>⚡ Elasticsearch</span>}
                    </div>
                  </div>

                  {result.intelligence?.summary && (
                    <div style={{ background:'#F8F4F0', borderLeft:'3px solid #D4908A', borderRadius:'0 10px 10px 0', padding:'12px 18px', marginBottom:18 }}>
                      <p style={{ fontFamily:"'Jost',sans-serif", margin:0, fontSize:13, fontWeight:300, color:'#2C2420', lineHeight:1.65 }}>
                        <span style={{ fontWeight:600, color:'#0F0D0C' }}>Research summary — </span>{result.intelligence.summary}
                      </p>
                    </div>
                  )}

                  {result.adapter_failures && result.adapter_failures.length > 0 && (
                    <div style={{ background:'#FDF4F4', border:'1px solid #F0D0CE', borderRadius:10, padding:'10px 16px', marginBottom:14 }}>
                      <p style={{ fontFamily:"'Jost',sans-serif", fontSize:11, fontWeight:600, color:'#C4615A', margin:'0 0 6px', textTransform:'uppercase', letterSpacing:'0.06em' }}>Partial results — some sources unavailable</p>
                      {result.adapter_failures.map((f:any,i:number) => {
                        const reason = f.reason as string; const src = f.source as string
                        const msg = reason==='auth_failed' ? `🔑 ${src} credentials missing`
                          : reason==='not_installed' ? `📦 ${src} package not installed`
                          : reason==='timeout' ? `⏱ ${src} timed out`
                          : `⚠ ${src} unavailable (${reason})`
                        return <p key={i} style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:12, color:'#C4615A', margin:i===0?0:'4px 0 0', lineHeight:1.5 }}>{msg}</p>
                      })}
                    </div>
                  )}

                  {result.results.length === 0 && (
                    <div style={{ background:'#F7F5F3', border:'1px solid #EAE4E2', borderRadius:12, padding:'20px 24px' }}>
                      <p style={{ fontFamily:"'Jost',sans-serif", fontSize:14, fontWeight:300, color:'#3D3532', margin:0, lineHeight:1.7 }}>
                        {result.agent_message || `No datasets matched "${result.query}". Try broader search terms.`}
                      </p>
                    </div>
                  )}

                  {result.results.map((ds,i) => <DatasetCard key={ds.dataset_id} ds={ds} insight={getInsight(ds)} idx={i} />)}
                </div>

                {/* Right panel */}
                {result.intelligence && (
                  <div style={{ width:224, flexShrink:0 }}>
                    {result.intelligence.follow_up_searches?.length > 0 && (
                      <div style={{ background:'#FAFAFA', border:'1px solid #EAE4E2', borderRadius:12, padding:'14px 16px', marginBottom:10 }}>
                        <p className="section-label">Related searches</p>
                        {result.intelligence.follow_up_searches.map((s:string,i:number) => (
                          <button key={i} onClick={() => doSearch(s)} style={{ display:'block', background:'none', border:'none', fontFamily:"'Jost',sans-serif", fontWeight:300, color:'#6B5E58', fontSize:12, cursor:'pointer', padding:'4px 0', textAlign:'left', lineHeight:1.5, width:'100%', transition:'color 0.12s' }}
                            onMouseEnter={e => (e.currentTarget.style.color='#0F0D0C')} onMouseLeave={e => (e.currentTarget.style.color='#6B5E58')}>
                            → {s}
                          </button>
                        ))}
                      </div>
                    )}
                    {result.intelligence.metadata_gaps?.length > 0 && (
                      <div style={{ background:'#FAFAFA', border:'1px solid #EAE4E2', borderRadius:12, padding:'14px 16px', marginBottom:10 }}>
                        <p className="section-label">Research gaps</p>
                        {result.intelligence.metadata_gaps.slice(0,4).map((g:string,i:number) => (
                          <p key={i} style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:12, color:'#B8893A', margin:'4px 0', lineHeight:1.5 }}>⚠ {g}</p>
                        ))}
                      </div>
                    )}
                    {result.intelligence.ecosystem_observation && (
                      <div style={{ background:'#FAFAFA', border:'1px solid #EAE4E2', borderRadius:12, padding:'14px 16px' }}>
                        <p className="section-label">Ecosystem note</p>
                        <p style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, fontSize:12, color:'#6B5E58', margin:0, lineHeight:1.6 }}>{result.intelligence.ecosystem_observation}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ── SEARCH BAR ── */}
          <div style={{ borderTop:'1px solid rgba(234,228,226,0.6)', padding:'14px 40px', background:'rgba(255,255,255,0.5)', backdropFilter:'blur(12px)', WebkitBackdropFilter:'blur(12px)' }}>
            <div className="search-wrap" style={{ maxWidth:860, margin:'0 auto' }}>
              <input ref={inputRef} className="search-input" value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key==='Enter' && doSearch(query)} placeholder="Describe the dataset you need…" />
              <button className="btn-primary" onClick={() => doSearch(query)} disabled={loading||!query.trim()}>{loading ? '…' : 'Search'}</button>
            </div>
            <p style={{ fontFamily:"'Jost',sans-serif", fontWeight:300, textAlign:'center', fontSize:9, color:'#D8CFC9', margin:'7px 0 0', letterSpacing:'0.08em', textTransform:'uppercase' }}>DataScout · HuggingFace · Kaggle · OpenML · Elasticsearch · Gemini AI</p>
          </div>
        </div>
      </div>
    </>
  )
}
