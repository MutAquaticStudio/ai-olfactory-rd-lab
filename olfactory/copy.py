"""Centralized product language for future localization."""

COPY = {
    "app_title": "Scent Molecule Studio",
    "app_subtitle": "Structure analysis and candidate design for fragrance R&D.",
    "analysis_tab": "Molecule analysis",
    "design_tab": "Candidate design",
    "analyze": "Analyze molecule",
    "predicted_profile": "Predicted odor profile",
    "target_profile": "Target odor profile",
    "sampling_diversity": "Sampling diversity",
    "generate": "Generate candidates",
    "candidate": "Candidate",
    "shortlist": "Shortlisted candidates",
    "target_fit": "Target fit",
    "sensory_profile": "Sensory profile",
    "target_descriptors": "Target descriptors",
    "supporting_descriptors": "Supporting descriptors",
    "local_new": "New to local reference set",
    "pubchem_not_found": "Not found in PubChem",
    "review_required": "Chemistry review required",
    "volatility_tier": "Estimated volatility tier",
    "volatility_caption": "MW-based estimate",
    "empty_design": (
        "Select one or more target descriptors, then generate a candidate set."
    ),
    "consent": (
        "I understand that candidate Isomeric SMILES and derived identifiers "
        "will be sent to the listed external reference providers."
    ),
    "reference_consent_required": (
        "Consent is required before candidate identifiers are sent to: {providers}."
    ),
    "status_title": "Generating and screening structures",
    "status_success": "Five screened candidates are ready for ranking.",
    "status_limit": (
        "The screening limit was reached before five candidates passed all checks."
    ),
    "pubchem_unavailable": (
        "PubChem verification is currently unavailable. "
        "Unverified structures were not shortlisted."
    ),
    "no_candidate": (
        "No structure passed all screening and identity checks in this run."
    ),
    "sampling": "Sampling structure",
    "invalid": "Invalid SMILES",
    "duplicate": "Duplicate in local reference set",
    "screen_failed": "Failed chemistry screen",
    "review": "Sent to chemistry review",
    "checking_pubchem": "Checking PubChem identity",
    "pubchem_found": "PubChem match found",
    "accepted": "Accepted for scoring",
    "ranking": "Ranking target fit",
    "preparing_3d": "Preparing 3D conformers",
    "stereo_enumeration": "Resolving stereoisomers",
    "stereo_review": "Stereo variants require chemistry review",
    "pubchem_unverified_log": "PubChem verification unavailable",
    "smiles_input": "SMILES",
    "smiles_help": "Stereochemical symbols such as @, @@, / and \\ are supported.",
    "analysis_empty": "Enter a SMILES string, then analyze the structure.",
    "invalid_smiles": "This SMILES string could not be parsed. Check the structure and try again.",
    "show_details": "Show details",
    "resource_error": "Application resources could not be loaded.",
    "analysis_error": "The molecule could not be analyzed.",
    "generation_error": "The candidate run could not be completed.",
    "structure_2d": "2D structure",
    "structure_3d": "Interactive 3D conformer",
    "stereo_caption": "Wedge and dash bonds show defined stereochemistry.",
    "conformer_unavailable": "3D conformer unavailable",
    "structural_ids": "Structural identifiers",
    "isomeric_smiles": "Isomeric SMILES",
    "canonical_smiles": "Canonical SMILES",
    "chemistry_screen": "Chemistry screen",
    "screen_pass": "Passed the configured screening profile",
    "screen_review": "Chemistry review required",
    "screen_reject": "Outside the configured screening profile",
    "prediction_disclaimer": (
        "Model predictions are not experimental sensory evidence. Category scores "
        "use a weighted-max projection of the 113 model outputs."
    ),
    "taxonomy_attribution": (
        "Osmo-compatible projection · The Osmo Scent Taxonomy v1.2, ODbL. "
        "No affiliation or endorsement is implied."
    ),
    "primary_facets": "Primary olfactory facets",
    "textures": "Olfactory textures",
    "sensations": "Sensations — incl. trigeminal",
    "chemesthesis": "Chemesthesis",
    "model_output": "Model output",
    "technical_details": "Technical details",
    "technical_model_text": (
        "OdorPredictor uses a 2,048-bit chiral Morgan fingerprint. "
        "SMILES_LSTM is used only for candidate sampling."
    ),
    "target_help": (
        "Target fit is the geometric mean of the selected descriptor probabilities."
    ),
    "diversity_help": (
        "Lower values favor familiar syntax; higher values increase variation and invalid samples."
    ),
    "consent_help": (
        "PubChem receives each screened candidate Isomeric SMILES. "
        "No request is made before consent."
    ),
    "review_queue": "Chemistry review queue",
    "review_queue_caption": (
        "These structures were not scored or shortlisted and require specialist review."
    ),
    "run_summary": "Run summary",
    "attempts": "Attempts",
    "accepted_count": "Accepted",
    "review_count": "Review",
    "unverified_count": "Unverified",
    "formula": "Formula",
    "exact_mw": "Exact MW",
    "log_p": "LogP",
    "tpsa": "TPSA",
    "rotatable_bonds": "Rotatable bonds",
    "heavy_atoms": "Heavy atoms",
    "sa_score": "SA score",
    "macrocycle_profile": "Macrocycle profile",
    "top_tier": "Top",
    "middle_tier": "Middle",
    "base_tier": "Base",
    "verification_caption": (
        "Reference checks report indexed identity or catalog evidence; they do not "
        "establish global novelty or patent clearance."
    ),
    "screening_caption": (
        "The chemistry screen is not a toxicity, IFRA, synthesis or experimental volatility assessment."
    ),
    "conformer_caption": (
        "Computational conformer ensemble — not an experimental structure."
    ),
    "radar_family_column": "Family",
    "radar_score_column": "Score",
    "descriptor_column": "Descriptor",
    "probability_column": "Probability",
    "macrocycle_ring": "{profile} · {size}-member ring",
    "event_line": "{attempt:03d} · {message} · {accepted}/{required} accepted",
    "candidate_heading": "{candidate} {rank}",
    "review_heading": "{index}. {label}",
    "family_chip": "{family} {score:.0f}%",
}


REASON_COPY = {
    "PARSE_OR_SANITIZE_ERROR": "The structure could not be sanitized",
    "MULTIPLE_FRAGMENTS": "Multiple molecular fragments",
    "NONZERO_FORMAL_CHARGE": "Non-zero formal charge",
    "RADICAL": "Radical electron detected",
    "UNSUPPORTED_ELEMENT": "Element outside the configured profile",
    "AROMATIC_NITRO": "Aromatic nitro alert",
    "PEROXIDE": "Peroxide O–O alert",
    "AZIDE": "Azide alert",
    "DIAZONIUM": "Diazonium alert",
    "AZO": "Azo alert",
    "HYDRAZINE": "Hydrazine alert",
    "ACYL_HALIDE": "Acyl halide alert",
    "ISOCYANATE": "Isocyanate alert",
    "SA_ABOVE_7": "Synthetic accessibility score above 7",
    "MW_OUT_OF_RANGE": "Exact MW outside the configured profile",
    "LOGP_OUT_OF_RANGE": "LogP outside the configured profile",
    "TPSA_OUT_OF_RANGE": "TPSA outside the configured profile",
    "ROTATABLE_BONDS_OUT_OF_RANGE": "Rotatable-bond count outside the configured profile",
    "HEAVY_ATOMS_OUT_OF_RANGE": "Heavy-atom count outside the configured profile",
    "SA_REVIEW_RANGE": "Synthetic accessibility score requires review",
    "HALOGEN_PRESENT": "Halogen present",
    "BRENK_OR_NIH_ALERT": "BRENK or NIH structural alert",
    "MW_NEAR_LIMIT": "Exact MW near the configured limit",
    "LOGP_NEAR_LIMIT": "LogP near the configured limit",
    "TPSA_NEAR_LIMIT": "TPSA near the configured limit",
    "ROTATABLE_BONDS_NEAR_LIMIT": "Rotatable-bond count near the configured limit",
    "PROFILE_ACCEPTED": "Within the configured screening profile",
    "STEREO_VARIANT_LIMIT": "More than four stereoisomers require specialist review",
    "CONFORMER_UNAVAILABLE": "No converged, validated 3D conformer was available",
    "KNOWN_IN_INDUSTRY_CATALOG": "Matching record found in a fragrance catalog",
    "KNOWN_IN_REFERENCE_SOURCE": "Matching record found in a reference source",
    "REFERENCE_UNVERIFIED": "Reference verification requires specialist review",
}
