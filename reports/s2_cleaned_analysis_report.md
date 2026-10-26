# Caption Analysis Report

**Generated:** 2025-10-15 10:57:36

**Source File:** `s2_cleaned.json`

---

## Dataset Overview

- **Total entries:** 105,723
- **With captions:** 105,723 (100.0%)
- **With references:** 105,723 (100.0%)
- **With fig_uri:** 0 (0.0%)

---

## Basic Statistics

### Caption Length (characters)

- **Min:** 18
- **Max:** 4,219
- **Average:** 390.0 (±318.8)
- **Median:** 302
- **25th percentile:** 161
- **75th percentile:** 523

#### Distribution

```
    18-438    │██████████████████████████████████████████████████ 70,887
   438-858    │██████████████████ 26,127
   858-1278   │████ 6,496
  1278-1698   │█ 1,603
  1698-2118   │ 422
  2118-2539   │ 129
  2539-2959   │ 43
  2959-3379   │ 8
  3379-3799   │ 4
  3799-4219   │ 4
```

### Word Count

- **Min:** 5
- **Max:** 644
- **Average:** 59.7 (±49.8)
- **Median:** 46
- **25th percentile:** 24
- **75th percentile:** 80

#### Distribution

```
     5-69     │██████████████████████████████████████████████████ 72,136
    69-133    │█████████████████ 24,943
   133-197    │████ 6,295
   197-261    │█ 1,690
   261-324    │ 438
   324-388    │ 149
   388-452    │ 45
   452-516    │ 15
   516-580    │ 8
   580-644    │ 4
```

### Sentence Count

- **Average:** 4.1
- **Single sentence:** 22,849 (21.6%)
- **Multiple sentences:** 79,832 (75.5%)

### Word Characteristics

- **Total unique words:** 261,164
- **Average word length:** 5.6 characters
- **Vocabulary richness:** 0.041

---

## Medical Term Analysis

### MODALITIES

**Coverage:** 93,158 captions (88.1%)

**Top terms:**

```
ct                        │████████████████████████████████████████ 76,907
mri                       │█████████ 17,594
ct scan                   │█████ 11,272
radiograph                │███ 7,082
x-ray                     │██ 5,443
ultrasound                │██ 5,196
microscopy                │ 1,719
sonography                │ 1,572
endoscopy                 │ 1,068
pathology                 │ 829
```

### ANATOMY

**Coverage:** 59,276 captions (56.1%)

**Top terms:**

```
cell                      │████████████████████████████████████████ 13,311
head                      │██████████████████████████████████ 11,399
brain                     │███████████████████████ 7,938
chest                     │██████████████████████ 7,497
tissue                    │█████████████████████ 7,098
artery                    │███████████████ 5,292
bone                      │██████████████ 4,859
lung                      │██████████████ 4,767
liver                     │███████████ 3,906
muscle                    │█████████ 3,126
```

### FINDINGS

**Coverage:** 42,852 captions (40.5%)

**Top terms:**

```
lesion                    │████████████████████████████████████████ 13,189
mass                      │███████████████████████████████████ 11,763
tumor                     │███████████████████████████████ 10,233
normal                    │███████████████████████████ 8,943
abnormal                  │█████████ 2,977
nodule                    │████████ 2,746
edema                     │███████ 2,543
fracture                  │██████ 2,131
cancer                    │██████ 2,062
hemorrhage                │████ 1,553
```

---

## Figure Type Classification

**Total captions:** 105,723

```
ct                        │██████████████████████████████████████████████████ 57.1%
mri                       │█████████████ 16.0%
x-ray                     │██████████ 11.6%
ultrasound                │██ 2.7%
chart                     │█ 1.8%
microscopy                │ 0.5%
unclassified              │█████████ 10.3%
```

---

## Punctuation Analysis

```
Ends With Period          │████████████████████████████████████████ 102,064
Ends With Question        │ 4
Ends With Exclamation     │ 2
Contains Parentheses      │███████████████████████████████ 80,913
Contains Brackets         │ 1,822
Contains Quotes           │██ 6,542
Contains Colon            │████████ 20,617
Contains Semicolon        │████████ 21,221
```

---

## Caption Quality Indicators

```
Good                      │████████████████ 24,411
Descriptive               │████████████████████████████████████████ 60,539
Technical                 │█████████████████████████████████ 50,783
Quantitative              │████████████████████ 31,568
```

---

## Problematic Patterns Detected

```
Excessive Caps            │████████████████████████████████████████ 105,713
Repeated Words            │ 1,102
Placeholder               │ 1
```

---

## Most Common Words

**Top 20 words (>3 characters):**

```
with                      │████████████████████████████████████████ 92,892
left                      │████████████████ 39,403
right                     │████████████████ 39,089
image                     │███████████████ 36,522
arrow                     │███████████████ 35,889
showing                   │███████████████ 35,615
images                    │████████████ 27,992
from                      │███████████ 26,978
after                     │███████████ 25,949
shows                     │██████████ 25,210
axial                     │██████████ 24,512
arrows                    │█████████ 21,888
cells                     │████████ 20,878
scan                      │████████ 20,587
weighted                  │████████ 20,120
tumor                     │███████ 17,675
were                      │███████ 16,570
patient                   │███████ 16,531
coronal                   │███████ 16,394
contrast                  │██████ 16,159
```

---

## Common Phrases (Bigrams)

**Top 15 two-word phrases:**

```
of the                    │████████████████████████████████████████ 104,683
in the                    │██████████████████████ 60,136
the right                 │██████ 17,789
to the                    │██████ 17,226
the left                  │██████ 17,210
at the                    │█████ 14,654
of a                      │█████ 13,889
and the                   │████ 13,022
on the                    │████ 12,851
in a                      │███ 9,823
ct scan                   │███ 9,623
with a                    │███ 9,515
from the                  │███ 9,498
with the                  │██ 7,505
(a) and                   │██ 7,135
```

---

## Reference Statistics

- **Total entries with references:** 105,723
- **Average references per entry:** 1.8
- **Min references:** 1
- **Max references:** 38
- **Median references:** 1

**Distribution:**

```
     1-5      │██████████████████████████████████████████████████ 99,281
     5-8      │██ 5,436
     8-12     │ 784
    12-16     │ 150
    16-20     │ 58
    20-23     │ 10
    23-27     │ 1
    27-31     │ 1
    31-34     │ 0
    34-38     │ 2
```

---

## Length Distribution

```
Very short (1-10)         │█████████ 5.5%
Short (11-30)             │██████████████████████████████████████████████ 27.6%
Medium (31-60)            │██████████████████████████████████████████████████ 29.5%
Long (61-100)             │████████████████████████████████████ 21.5%
Very long (101-200)       │███████████████████████ 13.9%
Extremely long (>200)     │███ 2.0%
```

---

## Quality Checks

```
Very short (< 5 words)    │ 0
Very long (> 100 words)   │███████ 16,878
Contains 'figure'         │█ 3,074
Starts with 'Figure'      │ 132
Incomplete sentences      │ 0
Contains numbers          │███████████████████████████████ 70,631
All lowercase             │ 81
All uppercase             │ 0
Proper capitalization     │████████████████████████████████████████ 89,178
```

---

## Data Completeness

- **Complete entries (caption + refs + hash):** 105,723 (100.0%)
- **Missing captions:** 0
- **Missing references:** 0
- **Missing pdf_hash:** 0

---

## Sample Captions

### Sample 1 (entry 0)

**Length:** 14 words

**Text:**

> Nuclear MR scan demonstrating the occipital lesion with irregular borders surrounded by slight edema.

### Sample 2 (entry 26430)

**Length:** 12 words

**Text:**

> Ultrasound of prostate showing enlarged prostate gland with estimated volume of 100g.

### Sample 3 (entry 52861)

**Length:** 26 words

**Text:**

> Spores of Microsporidium cypselurus sp. n.; transmission electron micrograph. Note 13-15 coils to the isofilar polar tube in the transverse section (inset). Scale bar 500 nm.

### Sample 4 (entry 79292)

**Length:** 101 words

**Text:**

> b-Catenin immunohistochemical staining of formalin-fixed, paraffin-embedded normal human colonic epithelia (panels A and B), and of a primary colorectal tumor (panels C and D) and a liver metastasis (panels E and F). The right panels contain magnifications ( Â 40) of specific areas from the left pan...
