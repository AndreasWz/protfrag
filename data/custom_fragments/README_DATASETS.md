# Custom Fragment Dataset Generation - Summary

## Overview

This project generates three distinct fragment datasets for protein fragment prediction, addressing the limitation that there are insufficient proteins with appropriate domain annotations to match the original SwissProt fragment distribution at scale.

**Note**: The fragment type `terminal_unannotated` has been excluded from all datasets as it represents noise rather than biologically meaningful fragments.

---

## Phase-by-Phase Breakdown

### Phase 1: Target Distribution Analysis
**Notebook**: `01_analyze_fragment_distribution.ipynb`

- Analyzes real SwissProt fragments (fragment=true)
- Extracts fragment type distribution and statistics
- Calculates target metrics for synthetic generation
- **Output**: Distribution summary, length statistics per fragment type

**Fragment Types** (excluding terminal_unannotated):
- `terminal_N`: N-terminal truncation
- `terminal_C`: C-terminal truncation
- `terminal_both`: Both termini truncated
- `internal_gap`: Internal gap with fusion
- `mixed`: Combination of terminal + internal gaps

---

### Phase 2: SwissProt Data Preparation
**Notebook**: `02_prepare_swissprot_data.ipynb`

- Loads full SwissProt database (~570K proteins)
- Filters for complete (non-fragment) proteins
- Extracts Domain annotations
- Categorizes proteins by fragment generation capability
- **Output**: Processed protein database, protein pools by fragment type

**Key Finding**: Insufficient proteins with correct domain annotations for all fragment types, particularly `terminal_both`.

---

## Three Dataset Generation Strategies

### Dataset 1: Maximum from Domain Annotations
**Notebook**: `03a_generate_dataset1_maximum.ipynb`

**Strategy**: Generate every possible fragment from every protein with domain annotations

**Characteristics**:
- Uses proteins with Domain annotations only
- Generates all possible fragment types from each protein
- No distribution matching - natural distribution based on annotation patterns
- Maximum data utilization
- Biologically grounded (uses real domain boundaries)

**Fragment Generation**:
- **terminal_N**: Remove domain at position 1, keep rest
- **terminal_C**: Remove domain at sequence end, keep rest
- **terminal_both**: Remove domains at both termini, keep middle
- **internal_gap**: Remove internal domain, fuse remaining parts
- **mixed**: Remove combination of terminal + internal domains

**Output Structure**:
```
source_accession | fragment_type | sequence | is_fragment | removed_region | sequence_length
```

**Pairing**: Each fragment paired with its complete source sequence

**Use Case**: Training on biologically realistic fragments with domain-level understanding

---

### Dataset 2: Augmented with Synthetic Cuts
**Notebook**: `03b_generate_dataset2_augmented.ipynb`

**Strategy**: Augment domain annotations with synthetic terminal cuts to create more fragment diversity

**Characteristics**:
- Starts with proteins that have domain annotations
- Adds synthetic N-terminal and/or C-terminal cut points where natural domains are missing
- Particularly increases `terminal_both` fragment availability
- Hybrid: mix of biological (domain-based) and synthetic cuts
- Labeled to distinguish synthetic vs domain-based

**Augmentation Rules**:
- If protein has only C-terminal domain → add synthetic N-terminal cut (10-30% from start)
- If protein has only N-terminal domain → add synthetic C-terminal cut (70-90% from end)
- If protein has only internal domains → add both synthetic terminal cuts

**Additional Column**: `is_synthetic` - indicates if fragment used synthetic cuts

**Output Structure**:
```
source_accession | fragment_type | sequence | is_fragment | removed_region | is_synthetic | sequence_length
```

**Comparison with Dataset 1**: Significantly more `terminal_both` fragments, higher overall count

**Use Case**: Training with enhanced diversity while maintaining some biological grounding

---

### Dataset 3: Distribution-Matched, Domain-Agnostic
**Notebook**: `03c_generate_dataset3_distribution_matched.ipynb`

**Strategy**: Completely ignore domain annotations, generate fragments using random cuts that match original distribution

**Characteristics**:
- Uses ALL complete SwissProt proteins (not limited to those with domain annotations)
- Exactly 100,000 fragments matching Phase 1 target distribution
- Exactly 200,000 total sequences (100K fragments + 100K complete)
- Length distributions matched to original fragments
- No dependency on domain annotations
- Fully synthetic cuts guided by statistical targets

**Fragment Generation** (domain-agnostic):
- **terminal_N**: Random N-terminal cut (10-30% range), target length sampled from original distribution
- **terminal_C**: Random C-terminal cut (70-90% range), target length sampled
- **terminal_both**: Random cuts at both termini, middle section kept
- **internal_gap**: Random internal portion removed, flanking regions fused
- **mixed**: Random combination of terminal + internal removals

**Distribution Matching**:
- Fragment type counts exactly match target percentages from Phase 1
- Fragment lengths sampled from original distribution per type
- Validates generated distribution against target

**Output Structure**:
```
source_accession | fragment_type | sequence | is_fragment | removed_region | generation_method | sequence_length
```

**Validation**: Includes comparison plots and statistics showing target vs generated distribution

**Use Case**: Maximum diversity dataset with perfect distribution control, ideal for balanced training

---

## Comparison Summary

| Feature | Dataset 1 | Dataset 2 | Dataset 3 |
|---------|-----------|-----------|-----------|
| **Basis** | Domain annotations | Augmented domains | Pure synthetic |
| **Size** | Variable (max possible) | Variable (larger than D1) | Fixed 200,000 |
| **Distribution** | Natural | Enhanced | Controlled match |
| **Protein Source** | With domains only | With domains only | All proteins |
| **Biological Grounding** | High | Medium | Low |
| **terminal_both Availability** | Low | High | Perfect match |
| **Diversity** | Limited by annotations | Enhanced | Maximum |
| **Use Case** | Biological realism | Hybrid approach | Balanced training |

---

## Files Generated

### Phase 1:
- `fragment_distribution_summary.csv` - Target statistics
- `fragment_type_distribution.png` - Visualization
- `overall_length_distribution.png` - Length analysis
- `length_distribution_by_type.png` - Per-type length analysis

### Phase 2:
- `swissprot_proteins_processed.csv` - All processed proteins
- `protein_pool_assignments.json` - Protein IDs by fragment type capability
- `protein_pool_summary.csv` - Pool statistics
- `domain_count_distribution.png` - Domain analysis
- `protein_pool_sizes.png` - Pool size visualization

### Dataset 1:
- `dataset1_maximum_fragments.csv` - Main dataset
- `dataset1_statistics.csv` - Summary statistics
- `dataset1_fragment_distribution.png` - Distribution plots
- `dataset1_length_distributions.png` - Length analysis

### Dataset 2:
- `dataset2_augmented_fragments.csv` - Main dataset
- `dataset2_analysis.png` - Comprehensive analysis
- Comparison with Dataset 1

### Dataset 3:
- `dataset3_distribution_matched.csv` - Main dataset
- `dataset3_validation.csv` - Distribution validation
- `dataset3_distribution_validation.png` - Target vs generated comparison
- `dataset3_length_distributions.png` - Length validation

---

## Common Dataset Columns

All datasets include:
- `source_accession`: UniProt accession ID
- `fragment_type`: Type of fragment or 'complete'
- `sequence`: Amino acid sequence
- `is_fragment`: Binary flag (1=fragment, 0=complete)
- `sequence_length`: Length in amino acids
- `removed_region`: Position ranges removed (empty for complete)

Dataset-specific columns:
- **Dataset 2**: `is_synthetic` - indicates synthetic cuts
- **Dataset 3**: `generation_method` - 'synthetic' or 'complete'

---

## Usage Recommendations

### For Training Binary Fragment Classifier (fragment vs complete):
- **Recommended**: Dataset 3 (balanced, controlled distribution)
- **Alternative**: Dataset 2 (good diversity with some biological grounding)

### For Understanding Biological Fragmentation Patterns:
- **Recommended**: Dataset 1 (domain-based, biologically realistic)
- **Alternative**: Dataset 2 (hybrid approach)

### For Maximum Training Data:
- **Recommended**: Dataset 1 (largest possible from annotations)
- **Alternative**: Combine Dataset 1 + Dataset 2 (even larger)

### For Specific Fragment Type Focus:
- **terminal_both**: Dataset 2 or Dataset 3 (Dataset 1 has insufficient)
- **Other types**: Any dataset works well

---

## Next Steps

**Phase 4**: Quality control and validation
- Cross-dataset comparison
- Amino acid composition analysis
- Sequence length distributions
- Duplicate detection
- Train/val/test splitting strategies

**Phase 5**: Export and documentation
- Final dataset selection
- Format conversions (FASTA, JSON, etc.)
- Complete documentation
- Usage examples

---

## Notes

1. All datasets exclude `terminal_unannotated` fragments (considered noise)
2. Each fragment is paired with its complete source sequence (1:1 ratio)
3. Sequences have minimum length threshold of 10 amino acids
4. All datasets are shuffled before saving
5. Random seed set to 42 for reproducibility

---

## Citation

If using these datasets, please acknowledge:
- SwissProt database (UniProt Consortium)
- Domain annotations from InterPro/Pfam
- This custom fragment generation pipeline

---

**Generated**: 2025-11-15
**Project**: ProtFrag - Protein Fragment Prediction