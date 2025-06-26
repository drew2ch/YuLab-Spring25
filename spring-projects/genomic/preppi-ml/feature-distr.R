# PPI Prediction Comparison with PrePPI (GO terms) Genomic ML Features: Cohen's d and Feature Distributions Study
# Andrew Chung, hc893; 6/6/2025

library(readxl)
library(tidyverse)
library(ggplot2)
library(effsize)

# import AF3 and PrePPI data sets
af3 = read.csv("ppi_features_max_avg.csv")
preppi = read_excel("Yilin_PrePPI_LRs_allclues.xlsx", sheet = "LRs_allclues")

# inner-merge by PPI pair, converting all "NULL" to 0.0.
data = merge(x = af3, y = preppi, by = "ppi", all = FALSE) %>% 
  filter(label > -1) %>%
  select(-ppi, -gene1, -gene2, -protein1, -protein2) %>%
  mutate(across(everything(), ~ case_when(
    . == "NULL" ~ 0.0, TRUE ~ as.numeric(.)
  ))) %>% select(label, everything()) %>%
  rename(max = `max(SM,PrP)`)
# 5 rows with non-finite values were discovered
data[is.infinite(as.matrix(data))] = NA
data = na.omit(data)
data$label = factor(data$label, levels = c(0,1))

# should have 5538 rows (unique protein dimers), 13 features, label
print(paste0("Dimensions of processed data set: ", paste(dim(data), collapse = ", ")))

# compute Cohen's d for each feature
feature.names = names(data)[-1]
cohens_d = setNames(rep(0, length(feature.names)), feature.names)

for (feature in feature.names){
  cd_res = cohen.d(
    formula = as.formula(paste0(feature, " ~ label")),
    data = data, hedges.correction = FALSE
  )
  cohens_d[feature] = abs(cd_res$estimate)
}

# print Cohen's d values
# d <= 0.2: no effect size
# 0.2 <= d <= 0.5: small effect size
# 0.5 <= d <= 0.8: moderate effect size
# 0.8 <= d: large effect size
for (feature in names(cohens_d)){
  d = round(cohens_d[feature], 3)
  print(paste0(feature, ": Cohen's d = ", d))
}

# Additional: plot side-by-side distributions
# this is quite challenging, as many features are plagued with high counts of 
# "NULL" (coerced to 0.0) that distorts the true distributions.
plot_density_by_label = function(data, feature_name) {
  ggplot(data, aes_string(x = feature_name, fill = "label")) +
    geom_density(alpha = 0.5) +
    labs(
      title = paste("Density of", feature_name, "by label"),
      x     = feature_name,
      y     = "Density",
      fill  = "Label"
    ) +
    theme_minimal()
}

# draw one plot per feature
for (feat in feature.names) {
  p = plot_density_by_label(data, feat)
  print(p)
}

