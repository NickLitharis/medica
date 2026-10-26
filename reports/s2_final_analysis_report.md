# Caption Analysis Report

**Generated:** 2025-10-15 10:33:47

**Source File:** `s2_final.json`

---

## Dataset Overview

- **Total entries:** 138,202
- **With captions:** 138,202 (100.0%)
- **With references:** 138,202 (100.0%)
- **With fig_uri:** 0 (0.0%)

---

## Basic Statistics

### Caption Length (characters)

- **Min:** 6
- **Max:** 6,407
- **Average:** 411.8 (±321.8)
- **Median:** 327
- **25th percentile:** 185
- **75th percentile:** 545

#### Distribution

```
     6-646    │██████████████████████████████████████████████████ 113,652
   646-1286   │█████████ 21,476
  1286-1926   │█ 2,614
  1926-2566   │ 379
  2566-3206   │ 61
  3206-3847   │ 16
  3847-4487   │ 3
  4487-5127   │ 0
  5127-5767   │ 0
  5767-6407   │ 1
```

### Word Count

- **Min:** 1
- **Max:** 953
- **Average:** 63.1 (±50.0)
- **Median:** 50
- **25th percentile:** 28
- **75th percentile:** 83

#### Distribution

```
     1-96     │██████████████████████████████████████████████████ 112,250
    96-191    │█████████ 22,344
   191-287    │█ 3,017
   287-382    │ 475
   382-477    │ 87
   477-572    │ 20
   572-667    │ 8
   667-763    │ 0
   763-858    │ 0
   858-953    │ 1
```

### Sentence Count

- **Average:** 5.2
- **Single sentence:** 12,092 (8.7%)
- **Multiple sentences:** 124,436 (90.0%)

### Word Characteristics

- **Total unique words:** 338,754
- **Average word length:** 5.5 characters
- **Vocabulary richness:** 0.039

---

## Medical Term Analysis

### MODALITIES

**Coverage:** 122,163 captions (88.4%)

**Top terms:**

```
ct                        │████████████████████████████████████████ 94,098
mri                       │████████ 19,120
computed tomography       │████████ 18,894
magnetic resonance        │████ 11,515
ct scan                   │████ 10,480
radiograph                │███ 9,133
x-ray                     │██ 6,653
ultrasound                │██ 6,002
microscopy                │ 2,258
sonography                │ 1,987
```

### ANATOMY

**Coverage:** 77,024 captions (55.7%)

**Top terms:**

```
cell                      │████████████████████████████████████████ 17,715
head                      │██████████████████████████████████ 15,071
brain                     │████████████████████████ 10,696
tissue                    │█████████████████████ 9,335
chest                     │███████████████████ 8,837
artery                    │███████████████ 6,759
bone                      │██████████████ 6,502
lung                      │█████████████ 5,908
liver                     │███████████ 5,098
muscle                    │█████████ 4,183
```

### FINDINGS

**Coverage:** 54,689 captions (39.6%)

**Top terms:**

```
lesion                    │████████████████████████████████████████ 17,137
mass                      │██████████████████████████████████ 14,738
tumor                     │███████████████████████████ 11,982
normal                    │███████████████████████████ 11,854
abnormal                  │█████████ 3,871
nodule                    │████████ 3,452
edema                     │███████ 3,333
fracture                  │██████ 2,924
cancer                    │██████ 2,807
hemorrhage                │████ 1,774
```

---

## Figure Type Classification

**Total captions:** 138,202

```
ct                        │██████████████████████████████████████████████████ 55.5%
mri                       │████████████████ 18.6%
x-ray                     │██████████ 11.2%
ultrasound                │██ 2.5%
chart                     │█ 1.8%
microscopy                │ 0.5%
unclassified              │█████████ 10.0%
```

---

## Punctuation Analysis

```
Ends With Period          │███████████████████████████████████████ 103,616
Ends With Question        │ 4
Ends With Exclamation     │ 2
Contains Parentheses      │████████████████████████████████████████ 105,742
Contains Brackets         │█ 3,831
Contains Quotes           │███ 8,686
Contains Colon            │██████████████ 39,606
Contains Semicolon        │██████████ 27,244
```

---

## Caption Quality Indicators

```
Good                      │████████████████ 32,158
Descriptive               │████████████████████████████████████████ 79,526
Technical                 │██████████████████████████████████ 67,887
Quantitative              │█████████████████████ 42,074
```

---

## Problematic Patterns Detected

```
Citation Markers          │ 1,072
Urls                      │ 1,330
Email                     │ 11
Excessive Caps            │████████████████████████████████████████ 137,876
Repeated Words            │ 716
Placeholder               │ 1
```

---

## Most Common Words

**Top 20 words (>3 characters):**

```
with                      │████████████████████████████████████████ 123,774
figure                    │██████████████████████████ 83,355
left                      │████████████████ 51,269
right                     │████████████████ 50,729
image                     │███████████████ 48,643
arrow                     │███████████████ 46,843
showing                   │██████████████ 46,226
images                    │████████████ 37,948
from                      │███████████ 36,328
after                     │██████████ 33,759
axial                     │██████████ 33,323
shows                     │██████████ 33,128
arrows                    │█████████ 28,550
cells                     │█████████ 27,859
weighted                  │████████ 26,689
tomography                │████████ 26,569
scan                      │████████ 25,877
imaging                   │████████ 25,212
patient                   │███████ 22,404
computed                  │███████ 22,336
```

---

## Common Phrases (Bigrams)

**Top 15 two-word phrases:**

```
of the                    │████████████████████████████████████████ 138,267
in the                    │██████████████████████ 78,113
to the                    │██████ 22,976
the right                 │██████ 22,665
the left                  │██████ 22,015
at the                    │█████ 19,267
of a                      │█████ 18,827
figure 1.                 │█████ 17,328
and the                   │████ 17,255
on the                    │████ 16,797
computed tomography       │████ 16,622
in a                      │███ 13,378
with a                    │███ 12,504
from the                  │███ 12,474
magnetic resonance        │███ 12,446
```

---

## Reference Statistics

- **Total entries with references:** 138,202
- **Average references per entry:** 1.9
- **Min references:** 1
- **Max references:** 38
- **Median references:** 1

**Distribution:**

```
     1-5      │██████████████████████████████████████████████████ 129,367
     5-8      │██ 7,432
     8-12     │ 1,088
    12-16     │ 204
    16-20     │ 84
    20-23     │ 19
    23-27     │ 5
    27-31     │ 1
    31-34     │ 0
    34-38     │ 2
```

---

## Length Distribution

```
Very short (1-10)         │█████ 3.3%
Short (11-30)             │███████████████████████████████████████ 24.9%
Medium (31-60)            │██████████████████████████████████████████████████ 31.5%
Long (61-100)             │████████████████████████████████████ 23.2%
Very long (101-200)       │███████████████████████ 15.0%
Extremely long (>200)     │███ 2.2%
```

---

## Quality Checks

```
Very short (< 5 words)    │ 1,010
Very long (> 100 words)   │██████ 23,719
Contains 'figure'         │███████████████████████ 80,085
Starts with 'Figure'      │██████████████████████ 78,620
Incomplete sentences      │█████████ 31,383
Contains numbers          │████████████████████████████████████████ 138,202
All lowercase             │ 0
All uppercase             │ 5
Proper capitalization     │████████████████████████████████████████ 138,202
```

---

## Data Completeness

- **Complete entries (caption + refs + hash):** 138,202 (100.0%)
- **Missing captions:** 0
- **Missing references:** 0
- **Missing pdf_hash:** 0

---

## Sample Captions

### Sample 1 (entry 0)

**Length:** 17 words

**Text:**

> Figure 4. Nuclear magnetic resonance scan demonstrating the occipital lesion with irregular borders surrounded by slight edema.

### Sample 2 (entry 34550)

**Length:** 4 words

**Text:**

> Figure 1. Intraoperative endoscopy

### Sample 3 (entry 69101)

**Length:** 92 words

**Text:**

> Fig 2. The two-step analysis. (A) Step 1: prospective assessment was done after 3-Tesla 2-dimensional-fluid-attenuated inversion recovery (3T-2D-FLAIR) or 7-Tesla 3-dimensional magnetization-transfer-weighted (7T-3D-MT w ) was coregistered to the imaging space of 7-Tesla 3-dimensional magnetization-...

### Sample 4 (entry 103651)

**Length:** 31 words

**Text:**

> Fig. 1. Contrast-enhanced magnetic resonance imaging: On post-contrast sections, slightly enhanced contrast uptake is noted on the thickened posterior wall of the bladder. Fig. 2. Cystoscopic image before oral steroid treatment....
