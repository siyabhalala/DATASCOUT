// datascout/frontend/src/lib/api.ts
// v3.5.0: expose all backend quality signals with correct types
// IMAGE:   per_class_stats, duplicate_count, blurry_count, corrupted_count,
//          dimension_info, num_classes
// TABULAR: columns_detail (for low-info / type-inconsistency detection),
//          duplicate_count (raw rows), null_columns with null_count primary

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// ── Request / Response Types ──────────────────────────────────────────────────

export interface ScoreBreakdown {
  task_relevance:    number
  quality:           number
  popularity:        number
  freshness:         number
  description_match: number
}

export interface BiasSignal {
  type:        string
  severity:    'low' | 'medium' | 'high'
  description: string
}

// ── IMAGE quality types ────────────────────────────────────────────────────────

/** Per-class image quality stats — all counts are RAW integers */
export interface PerClassStat {
  name:       string
  images:     number   // real folder count
  duplicates: number   // images whose hash appears elsewhere in the dataset
  blurry:     number   // images below blur threshold (sampled)
  corrupted:  number   // images PIL/cv2 could not open (sampled)
}

export interface DimensionInfo {
  is_consistent:    boolean | null
  dominant_size:    [number, number] | null   // [width, height]
  unique_sizes:     number
  undersized_count: number   // images < 32×32
  oversized_count:  number
  warning:          string | null
}

// ── TABULAR quality types ─────────────────────────────────────────────────────

/** Column with >5% missing values — null_count is primary signal */
export interface NullColumn {
  name:        string
  null_count:  number   // RAW count — primary
  missing_pct: number   // percentage — secondary
}

/** Full column detail — used for low-info / type-inconsistency detection */
export interface ColumnDetail {
  name:             string
  type:             string   // "numeric" | "categorical" | "text" | "boolean" | "id" | "mixed" | "unknown"
  null_count:       number
  missing_pct:      number
  unique_count:     number
  total_count:      number
  cardinality_ratio: number  // 0-1 (unique/total)
  is_id:            boolean
  is_target:        boolean
  top_categories:   [string, number][] | null  // [(value, count), ...]
}

export interface ColumnSummary {
  total:       number | null
  numeric:     number | null
  categorical: number | null
  text:        number | null
  missing_pct: number | null
}

// ── Shared types ──────────────────────────────────────────────────────────────

export interface GeographicBias {
  detected: boolean
  regions:  string[]
  warning:  string | null
}

// ── Deep analysis composite ───────────────────────────────────────────────────

export interface DeepAnalysis {
  quality_score:      number | null
  completeness_score: number | null
  balance_score:      number | null
  uniqueness_score:   number | null
  suggested_fixes:    string[]
  is_partial:         boolean

  // ── IMAGE ──────────────────────────────────────────────────────────────────
  class_distribution: Record<string, number> | null   // {class: count}
  num_classes:        number | null
  total_images:       number | null                   // sum of all class images (real)
  total_images_est:   number | null                   // from description text (no auth)
  num_classes_est:    number | null                   // from description text (no auth)
  analysis_error:     string | null                   // why download/listing failed
  per_class_stats:    PerClassStat[] | null            // per-class raw counts

  // Aggregate image quality signals
  duplicate_pct:   number | null   // % of images that are duplicates
  duplicate_count: number | null   // raw duplicate image count
  blur_pct:        number | null   // % blurry (sampled)
  blurry_count:    number | null   // raw blurry count (from sample)
  corrupted_pct:   number | null   // % corrupted (sampled)
  corrupted_count: number | null   // raw corrupted count (from sample)
  dimension_info:  DimensionInfo | null

  // ── TABULAR ────────────────────────────────────────────────────────────────
  null_columns:    NullColumn[]        // columns with >5% missing
  columns_detail:  ColumnDetail[]      // all columns for deeper analysis
  columns:         ColumnSummary | null

  // ── SHARED ─────────────────────────────────────────────────────────────────
  geographic_bias: GeographicBias
}

// ── Dataset result ────────────────────────────────────────────────────────────

export interface DatasetResult {
  rank:                   number
  dataset_id:             string
  title:                  string
  description:            string
  source:                 string
  source_url:             string
  tags:                   string[]
  task_types:             string[]
  row_count:              number | null
  column_count:           number | null
  license:                string | null
  last_updated:           string | null
  download_count:         number | null
  author:                 string | null
  metadata_completeness:  number
  has_description:        boolean
  has_schema_info:        boolean
  has_license_info:       boolean
  composite_score:        number
  score_breakdown:        ScoreBreakdown
  why_ranked_here:        string
  strengths:              string[]
  weaknesses:             string[]
  score_labels:           Record<string, string>
  quality_tier:           string
  risk_indicators:        string[]
  bias_signals:           BiasSignal[]
  bias_warnings:          string[]
  missing_fields:         string[]
  risk_level:             'low' | 'medium' | 'high'
  task_match_type:        string
  task_match_explanation: string
  freshness_explanation:  string
  days_since_update:      number | null
  annotation_score:       number | null
  class_imbalance_detected: boolean
  missing_values_detected:  boolean
  deep_analysis:          DeepAnalysis | null
}

export interface AdapterFailure {
  source: string
  reason: 'auth_failed' | 'timeout' | 'rate_limited' | 'not_installed' | 'unknown'
}

export interface SearchResponse {
  query:                  string
  results:                DatasetResult[]
  total_found:            number
  returned:               number
  confidence:             string
  intelligence_available: boolean
  enriched_query?:        string
  domain_detected?:       string
  retrieval_method?:      string
  agent_message?:         string
  sources_used?:          string[]
  adapter_failures?:      AdapterFailure[]
  broadened?:             boolean
  query_used?:            string
  intelligence?: {
    summary:               string
    dataset_insights:      any[]
    metadata_gaps:         string[]
    follow_up_searches:    string[]
    ecosystem_observation: string
    infrastructure_notes?: string[]
  }
  error_message?:         string
  processing_time_ms?:    number
  request_id?:            string
}

export interface SearchSession {
  id:          string
  query:       string
  timestamp:   Date
  resultCount: number
}

// ── Raw backend shape ─────────────────────────────────────────────────────────

interface _BackendQuality {
  completeness:        number
  consistency:         number
  validity:            number
  composite:           number
  issues:              string[]
  deep_quality_score:  number | null
  completeness_score:  number | null
  balance_score:       number | null
  uniqueness_score:    number | null
  suggested_fixes:     string[]
  analysis_partial:    boolean
  // image
  class_distribution:  Record<string, number> | null
  num_classes:         number | null
  total_images:        number | null
  per_class_stats:     PerClassStat[] | null
  duplicate_pct:       number | null
  duplicate_count:     number | null
  blur_pct:            number | null
  blurry_count:        number | null
  corrupted_pct:       number | null
  corrupted_count:     number | null
  dimension_info:      DimensionInfo | null
  total_images_est:    number | null
  num_classes_est:     number | null
  analysis_error:      string | null
  // tabular
  null_columns:        NullColumn[]
  columns_detail:      ColumnDetail[]
  columns_summary:     {
    total: number | null; numeric: number | null; categorical: number | null
    text: number | null; missing_pct: number | null
  } | null
  // shared
  geographic_bias: { detected: boolean; regions: string[]; warning: string | null }
}

interface _BackendDataset {
  rank:                  number
  canonical_id:          string
  title:                 string
  description:           string
  source:                string
  source_url:            string
  tags:                  string[]
  task_types:            string[]
  row_count:             number | null
  column_count:          number | null
  column_names:          string[] | null
  license_type:          string | null
  last_updated:          string | null
  download_count:        number | null
  author:                string | null
  metadata_completeness: number
  composite_score:       number
  score_breakdown:       ScoreBreakdown
  why_ranked_here:       string
  strengths:             string[]
  weaknesses:            string[]
  bias_warnings:         string[]
  score_labels:          Record<string, string>
  quality_tier:          string
  freshness_days:        number | null
  freshness_explanation: string
  analysis?: {
    quality:      _BackendQuality
    summary:      any
    target_column: any
  }
}

interface _BackendResponse {
  query:                  string
  results:                _BackendDataset[]
  total_found:            number
  returned:               number
  confidence:             string
  intelligence_available: boolean
  expanded_query?:        string
  agent_message?:         string
  sources_used?:          string[]
  adapter_failures?:      AdapterFailure[]
  broadened?:             boolean
  query_used?:            string
  processing_time_ms?:    number
  request_id?:            string
  retrieval_method?:      string
  intelligence?:          any
}

// ── Mapper ────────────────────────────────────────────────────────────────────

function mapDataset(raw: _BackendDataset): DatasetResult {
  const q = raw.analysis?.quality ?? null

  let deep_analysis: DeepAnalysis | null = null

  if (q) {
    deep_analysis = {
      quality_score:      q.deep_quality_score   ?? null,
      completeness_score: q.completeness_score   ?? null,
      balance_score:      q.balance_score        ?? null,
      uniqueness_score:   q.uniqueness_score     ?? null,
      suggested_fixes:    q.suggested_fixes      ?? [],
      is_partial:         q.analysis_partial     ?? false,

      // image
      class_distribution: q.class_distribution   ?? null,
      num_classes:        q.num_classes           ?? null,
      total_images:       q.total_images          ?? null,
      total_images_est:   q.total_images_est      ?? null,
      num_classes_est:    q.num_classes_est       ?? null,
      analysis_error:     q.analysis_error        ?? null,
      per_class_stats:    q.per_class_stats        ?? null,
      duplicate_pct:      q.duplicate_pct         ?? null,
      duplicate_count:    q.duplicate_count        ?? null,
      blur_pct:           q.blur_pct              ?? null,
      blurry_count:       q.blurry_count          ?? null,
      corrupted_pct:      q.corrupted_pct         ?? null,
      corrupted_count:    q.corrupted_count        ?? null,
      dimension_info:     q.dimension_info         ?? null,

      // tabular
      null_columns:   (q.null_columns   ?? []).map((c: any) => ({
        name:        c.name,
        null_count:  c.null_count  ?? null,
        missing_pct: c.missing_pct ?? 0,
      })),
      columns_detail: (q.columns_detail ?? []).map((c: any) => ({
        name:             c.name,
        type:             c.type,
        null_count:       c.null_count       ?? 0,
        missing_pct:      c.missing_pct      ?? 0,
        unique_count:     c.unique_count     ?? 0,
        total_count:      c.total_count      ?? 0,
        cardinality_ratio: c.cardinality_ratio ?? 0,
        is_id:            c.is_id            ?? false,
        is_target:        c.is_target        ?? false,
        top_categories:   c.top_categories   ?? null,
      })),
      columns: q.columns_summary ? {
        total:       q.columns_summary.total       ?? null,
        numeric:     q.columns_summary.numeric     ?? null,
        categorical: q.columns_summary.categorical ?? null,
        text:        q.columns_summary.text        ?? null,
        missing_pct: q.columns_summary.missing_pct ?? null,
      } : null,

      // shared
      geographic_bias: q.geographic_bias
        ? { detected: q.geographic_bias.detected, regions: q.geographic_bias.regions ?? [], warning: q.geographic_bias.warning ?? null }
        : { detected: false, regions: [], warning: null },
    }
  }

  const licenseRaw = raw.license_type
  return {
    rank:                    raw.rank,
    dataset_id:              raw.canonical_id,
    title:                   raw.title,
    description:             raw.description,
    source:                  raw.source,
    source_url:              raw.source_url,
    tags:                    raw.tags            ?? [],
    task_types:              raw.task_types      ?? [],
    row_count:               raw.row_count       ?? null,
    column_count:            raw.column_count    ?? null,
    license:                 licenseRaw          ?? null,
    last_updated:            raw.last_updated    ?? null,
    download_count:          raw.download_count  ?? null,
    author:                  raw.author          ?? null,
    metadata_completeness:   raw.metadata_completeness ?? 0,
    has_description:         (raw.description?.length ?? 0) > 20,
    has_schema_info:         (raw.column_names?.length ?? raw.column_count ?? 0) > 0,
    has_license_info:        licenseRaw != null,
    composite_score:         raw.composite_score,
    score_breakdown:         raw.score_breakdown ?? { task_relevance:0, quality:0, popularity:0, freshness:0, description_match:0 },
    why_ranked_here:         raw.why_ranked_here ?? '',
    strengths:               raw.strengths       ?? [],
    weaknesses:              raw.weaknesses      ?? [],
    score_labels:            raw.score_labels    ?? {},
    quality_tier:            raw.quality_tier    ?? 'incomplete',
    risk_indicators:         raw.bias_warnings   ?? [],
    bias_signals:            [],
    bias_warnings:           raw.bias_warnings   ?? [],
    missing_fields:          q?.issues           ?? [],
    risk_level:              _riskLevel(raw.bias_warnings ?? [], raw.weaknesses ?? []),
    task_match_type:         '',
    task_match_explanation:  '',
    freshness_explanation:   raw.freshness_explanation ?? '',
    days_since_update:       raw.freshness_days  ?? null,
    annotation_score:        null,
    class_imbalance_detected: (raw.bias_warnings ?? []).some(w => w.toLowerCase().includes('imbalance')),
    missing_values_detected:  (q?.issues ?? []).some(i => i.toLowerCase().includes('missing')),
    deep_analysis,
  }
}

function _riskLevel(biasWarnings: string[], weaknesses: string[]): 'low' | 'medium' | 'high' {
  if (biasWarnings.length >= 2) return 'high'
  if (biasWarnings.length === 1 || weaknesses.length >= 3) return 'medium'
  return 'low'
}

function mapResponse(raw: _BackendResponse): SearchResponse {
  return {
    query:                   raw.query,
    results:                 (raw.results ?? []).map(mapDataset),
    total_found:             raw.total_found,
    returned:                raw.returned,
    confidence:              raw.confidence,
    intelligence_available:  raw.intelligence_available,
    enriched_query:          raw.expanded_query,
    domain_detected:         undefined,
    retrieval_method:        raw.retrieval_method,
    agent_message:           raw.agent_message,
    sources_used:            raw.sources_used,
    adapter_failures:        raw.adapter_failures,
    broadened:               raw.broadened,
    query_used:              raw.query_used,
    intelligence: raw.intelligence ?? (
      (raw.results?.length ?? 0) > 0 ? {
        summary: `Found ${raw.returned} datasets from ${raw.total_found} candidates.`,
        dataset_insights: (raw.results ?? []).map(ds => ({
          dataset_id:      ds.canonical_id,
          why_ranked:      ds.why_ranked_here ?? '',
          strengths:       ds.strengths       ?? [],
          weaknesses:      ds.weaknesses      ?? [],
          score_narrative: ds.why_ranked_here ?? '',
        })),
        metadata_gaps:         [],
        follow_up_searches:    [],
        ecosystem_observation: '',
      } : undefined
    ),
    error_message:       undefined,
    processing_time_ms:  raw.processing_time_ms,
    request_id:          raw.request_id,
  }
}

// ── Public API functions ──────────────────────────────────────────────────────

export async function searchDatasets(
  query:          string,
  maxResults    = 10,
  sources       = ['huggingface', 'openml', 'kaggle'],
  previousQuery?: string,
): Promise<SearchResponse> {
  const body: Record<string, any> = { raw_query: query, max_results: maxResults }
  if (previousQuery) body.previous_query = previousQuery

  const res = await fetch(`${API_BASE}/api/v2/search`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })

  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}))
    throw new Error(errBody?.detail ?? errBody?.message ?? `HTTP ${res.status}`)
  }

  return mapResponse(await res.json() as _BackendResponse)
}

export async function checkHealth(): Promise<{ status: string; elasticsearch: string; embedding_engine: string }> {
  const res = await fetch(`${API_BASE}/api/v2/health`)
  if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`)
  return res.json()
}
