export type ThemePreference = 'system' | 'light' | 'dark';

export interface NamedProbability {
  name: string;
  probability: number;
}

export interface TaxonomyProfile {
  facets: NamedProbability[];
  textures: NamedProbability[];
  sensations: NamedProbability[];
  projection_name: string;
  taxonomy_version: string;
}

export interface ChemicalScreen {
  decision: 'PASS' | 'REVIEW' | 'REJECT';
  reason_codes: string[];
  reasons: string[];
  descriptors: Record<string, number>;
  is_macrocycle: boolean;
  macrocycle_ring_size: number | null;
  macrocycle_carbon_fraction: number;
  macrocycle_heteroatoms: number;
  alerts: string[];
}

export interface ConformerRecord {
  molblock: string;
  relative_energy: number;
}

export interface ConformerEnsemble {
  available: boolean;
  method: string | null;
  requested_count: number;
  embedded_count: number;
  converged_count: number;
  is_macrocycle: boolean;
  error: string | null;
  conformers: ConformerRecord[];
}

export interface DisplayDescriptors {
  formula: string;
  exact_mw: number;
  log_p: number;
  tpsa: number;
  rotatable_bonds: number;
  heavy_atoms: number;
  sa_score: number;
  estimated_volatility_tier: string;
  volatility_basis: string;
}

export interface PredictedOdorProfile {
  top_descriptors: NamedProbability[];
  taxonomy: TaxonomyProfile;
  model_output: NamedProbability[];
}

export interface StereoOption {
  isomeric_smiles: string;
  cip_summary: string;
  structure_2d_svg: string;
}

export type ReferenceStatus = 'MATCH' | 'NO_MATCH' | 'AMBIGUOUS' | 'UNVERIFIED' | 'NOT_CONFIGURED';
export type ReferenceMatchLevel = 'EXACT_STEREO' | 'EXACT_CONNECTIVITY' | 'EXACT_CAS' | 'NAME_ONLY';
export type ReferenceGateStatus = 'PASS' | 'BLOCKED_MATCH' | 'REVIEW_REQUIRED' | 'NOT_RUN';

export interface ReferenceEvidence {
  provider: string;
  status: ReferenceStatus;
  match_level: ReferenceMatchLevel | null;
  queried_identifier: string | null;
  record_ids: string[];
  record_urls: string[];
  checked_at: string | null;
  source_version: string | null;
  error_code: string | null;
}

export interface ReferenceGate {
  status: ReferenceGateStatus;
  blocking_providers: string[];
  reason_code: string | null;
}

export interface ReferenceProviderMeta {
  provider: string;
  display_name: string;
  source_type: 'STRUCTURAL_IDENTITY' | 'FRAGRANCE_CATALOG';
  enabled: boolean;
  external: boolean;
  query_types: string[];
  dataset_version: string | null;
  license_status: string;
  configuration_error: string | null;
}

interface AnalysisBase {
  input_smiles: string;
  identifiers: { isomeric_smiles: string; canonical_smiles: string };
  structure_2d_svg: string;
  chemistry_screen: ChemicalScreen;
  display_descriptors: DisplayDescriptors;
  unresolved_stereo_elements: number;
  reference_checks: ReferenceEvidence[];
  reference_gate: ReferenceGate;
}

export interface CompleteAnalysis extends AnalysisBase {
  analysis_state: 'COMPLETE';
  stereo_options: [];
  predicted_odor_profile: PredictedOdorProfile;
  prediction_v2: PredictionV2;
  conformer_ensemble: ConformerEnsemble;
}

export interface StereoRequiredAnalysis extends AnalysisBase {
  analysis_state: 'STEREO_REQUIRED';
  stereo_options: StereoOption[];
  predicted_odor_profile: null;
  prediction_v2: null;
  conformer_ensemble: null;
}

export interface StereoInputRequiredAnalysis extends AnalysisBase {
  analysis_state: 'STEREO_INPUT_REQUIRED';
  stereo_options: [];
  predicted_odor_profile: null;
  prediction_v2: null;
  conformer_ensemble: null;
}

export type AnalysisResult = CompleteAnalysis | StereoRequiredAnalysis | StereoInputRequiredAnalysis;

export interface AppMeta {
  label_names: string[];
  taxonomy_version: string;
  projection_name: string;
  generation_limits: {
    required_candidates: number;
    shortlist_count: number;
    max_attempts: number;
    max_seconds: number;
    max_event_lines: number;
    candidate_stereo_limit: number;
  };
  conformer_ensemble: {
    normal_sampling_count: number;
    macrocycle_sampling_count: number;
    max_displayed: number;
    normal_cluster_rmsd: number;
    macrocycle_cluster_rmsd: number;
    cache_size: number;
  };
  stereo: { analysis_option_limit: number; candidate_variant_limit: number };
  capabilities: { structure_2d: boolean; conformer_3d: boolean };
  data_foundation: {
    available: boolean;
    label_semantics: PresenceState[];
    intensity_scale: [number, number];
  };
  reference_verification: {
    providers: ReferenceProviderMeta[];
    required_external_consents: string[];
    shortlist_policy: string;
  };
  models: Record<string, Record<string, unknown>>;
}

export type PresenceState = 'PRESENT' | 'ABSENT' | 'UNASSESSED';

export interface PredictionV2 {
  model_version: string;
  dataset_version: string;
  calibration_version: string;
  model_status: string;
  calibrated: boolean;
  nearest_training_similarity: number | null;
  reliability_state: 'IN_DOMAIN' | 'LIMITED_EVIDENCE' | 'OUT_OF_DOMAIN';
  presence_predictions: Array<NamedProbability & {
    expected_intensity: number | null;
    uncertainty: number | null;
    decision_threshold: number | null;
  }>;
  limitations: string[];
  /** Additive batch-contract fields; v1 consumers continue using the list above. */
  presence_probability?: number[];
  expected_intensity?: Array<number | null>;
  ensemble_uncertainty?: Array<number | null>;
  training_similarity?: number | null;
  reliability?: 'IN_DOMAIN' | 'LIMITED_EVIDENCE' | 'OUT_OF_DOMAIN';
}

export interface AssessmentPayload {
  study_name: string;
  session_name: string;
  assessor_id: string;
  blinded_sample_code: string;
  smiles: string;
  descriptor: string;
  presence_state: PresenceState;
  concentration: number;
  concentration_unit: string;
  solvent: string;
  temperature_c: number;
  confidence: number;
  replicate_number: number;
  intensity: number | null;
  source_name: string;
  source_version: string;
  source_license: string;
  preparation_time_minutes: number | null;
  notes: string | null;
}

export interface ImportIssue {
  row: number;
  field: string;
  code: string;
  message: string;
  severity: 'ERROR' | 'WARNING';
}

export interface ImportValidation {
  filename: string;
  sha256: string;
  row_count: number;
  valid_count: number;
  is_valid: boolean;
  validation_token: string | null;
  issues: ImportIssue[];
  preview: Array<Record<string, unknown>>;
}

export interface DatasetVersion {
  dataset_version: string;
  created_at: string;
  row_count: number;
  sha256: string;
  parquet_path?: string;
  manifest_path?: string;
}

export type GenerationPhase =
  | 'SAMPLING'
  | 'INVALID'
  | 'DUPLICATE'
  | 'REJECTED'
  | 'REVIEW'
  | 'PUBCHEM_CHECK'
  | 'PUBCHEM_FOUND'
  | 'PUBCHEM_UNVERIFIED'
  | 'CHECKING_REFERENCES'
  | 'CATALOG_MATCH'
  | 'REFERENCE_UNVERIFIED'
  | 'REFERENCE_ACCEPTED'
  | 'ACCEPTED'
  | 'STEREO_ENUMERATION'
  | 'STEREO_REVIEW'
  | 'RANKING'
  | 'PREPARING_3D';

export interface GenerationEvent {
  phase: GenerationPhase;
  attempt: number;
  accepted: number;
  invalid: number;
  duplicates: number;
  rejected: number;
  reviews: number;
  found: number;
  unverified: number;
  reference_matches: number;
  reference_unverified: number;
  detail: string | null;
}

export interface RankedCandidate {
  isomeric_smiles: string;
  canonical_smiles: string;
  target_fit: number;
  target_probabilities: NamedProbability[];
  supporting_descriptors: NamedProbability[];
  structure_2d_svg: string;
  conformer_ensemble: ConformerEnsemble;
  chemistry_screen: ChemicalScreen;
  display_descriptors: DisplayDescriptors;
  novelty: { status: string; cids: number[]; error_code: string | null };
  reference_checks: ReferenceEvidence[];
  reference_gate: ReferenceGate;
}

export interface ReviewCandidate {
  isomeric_smiles: string;
  structure_2d_svg: string | null;
  chemistry_screen: ChemicalScreen;
  review_category: 'CHEMISTRY' | 'REFERENCE';
  reference_checks: ReferenceEvidence[];
  reference_gate: ReferenceGate;
}

export interface GenerationComplete {
  shortlist: RankedCandidate[];
  review_queue: ReviewCandidate[];
  summary: {
    attempts: number;
    accepted: number;
    reviews: number;
    invalid: number;
    duplicates: number;
    rejected: number;
    found: number;
    unverified: number;
    reference_matches: number;
    reference_unverified: number;
    elapsed_seconds: number;
    reached_attempt_limit: boolean;
    reached_time_limit: boolean;
  };
}

export interface ProductError {
  code: string;
  message: string;
  technical_details?: string;
}
