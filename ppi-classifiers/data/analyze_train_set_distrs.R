# Andrew Chung (hc893), 8/22/2025

library(dplyr)
library(tidyverse)
library(readxl)

PWD = "C:/Users/hychu/OneDrive/Desktop/Summer25/github/ppi-classifiers/data/"
train = read.csv(paste0(PWD, "train_set.csv"))
output_file = paste0(PWD, "train_set_dists.txt")
structural = train %>% filter(str_label > -1)
non_structural = train %>% filter(non_struct_label > -1)

# structural, non-structural Positive/Negative count
cat(
  sprintf(
    "Structural (Total %d): %d P, %d N",
    nrow(structural),
    sum(structural$str_label == 1),
    sum(structural$str_label == 0)
  ),  "\n",
  sprintf(
    "Non-Structural (Total %d): %d P, %d N",
    nrow(non_structural),
    sum(non_structural$non_struct_label == 1),
    sum(non_structural$non_struct_label == 0)
  ),
  file = output_file
)
cat( "\n============\n", file = output_file, append = TRUE)
# homodimer count for each
structural = structural %>%
  mutate(p1 = sapply(strsplit(ppi, ':'), `[`, 1), p2 = sapply(strsplit(ppi, ':'), `[`, 2))
non_structural = non_structural %>%
  mutate(p1 = sapply(strsplit(ppi, ':'), `[`, 1), p2 = sapply(strsplit(ppi, ':'), `[`, 2))
cat(
  sprintf("Structural: %d Homodimers", sum(structural$p1 == structural$p2)), "\n",
  sprintf("Non-Structural: %d Homodimers", sum(non_structural$p1 == non_structural$p2)),
  file = output_file,
  append = TRUE
)

# ADDITIONAL: mark PRePPI-derived query pairs for test set partition
preppi = read_excel(paste0(PWD, "Yilin_PrePPI_LRs_allclues.xlsx"), sheet = "LRs_allclues")
preppi.pairs = preppi$ppi
train = train %>%
  mutate(is_preppi = if_else(ppi %in% preppi.pairs, 1, 0)) %>%
  left_join(preppi %>% select(ppi, Total), by = 'ppi') %>%
  mutate(Total = replace_na(Total, 0))
# apparently, some (a fringe minority) of PrePPI pairs contain 'Total' values of infinity.
if (any(is.infinite(train$Total))){
  infinite_rows <- which(is.infinite(train$Total))
  cat(paste('The \'Total\' column contains infinite values at positions', infinite_rows, '\n'))
  for (row in infinite_rows){
    train$is_preppi[row] = 0
    train$Total[row] = 0
  }
}
write.csv(train, paste0(PWD, "train_set_with_preppi.csv"), row.names = FALSE)
