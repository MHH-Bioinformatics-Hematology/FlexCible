Verbatim copies (registry section 73), made so that the original working trees ~/src/C.Origami and
~/src/epiphany stay untouched:
  corigami_blocks.py, corigami_models.py   <- C.Origami src/corigami/model/{blocks,corigami_models}.py
                                              (one import line changed)
  epiphany_model_10kb.py                   <- epiphany/epiphany/model_10kb.py (Net, Disc; the generator
                                              of adversarial.py, no final ReLU)
  centrotelo_hg38.bed                      <- C.Origami corigami_data/data/hg38/centrotelo.bed
  (the two torchvision import lines of epiphany_model_10kb.py were removed; unused)
