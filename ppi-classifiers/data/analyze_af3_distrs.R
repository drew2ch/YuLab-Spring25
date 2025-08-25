# Andrew Chung (hc893), 8/14/2025

library(ggplot2)
library(dplyr)
library(tidyverse)

PWD = "C:/Users/hychu/OneDrive/Desktop/Summer25/github/spring-projects/data/"
preppi = read.csv(paste0(PWD, "final_af3_results.csv"))
plot_data_non_struct = preppi %>%
  group_by(non_struct_label) %>%
  summarise(non_struct_count = n(), .groups = 'drop') %>%
  rename(label = non_struct_label)
plot_data_struct = preppi %>%
  group_by(str_label) %>%
  summarise(struct_count = n(), .groups = 'drop') %>%
  rename(label = str_label)
plot_data = merge(
  as.data.frame(plot_data_non_struct), 
  as.data.frame(plot_data_struct)
) %>%
  pivot_longer(
    cols = c(non_struct_count, struct_count), 
    names_to = 'count_type', values_to = 'count'
) %>% mutate(count_type = str_replace(count_type, '_count', '') %>% str_to_title()) %>%
  filter()

ggplot(
  filter(plot_data, label %in% c(0, 1)), 
  aes(x = count_type, y = count, fill = as.factor(label))
) +
  geom_col(position = 'dodge') +
  geom_text(
    aes(label = count),
    position = position_dodge(width = 0.9),
    vjust = -0.5
  ) +
  labs(
    title = "Structural vs. Nonstructural Distributions (AF3)",
    x = 'Structure Type', y = 'Count', fill = 'Label'
  ) +
  scale_x_discrete(labels = c("Non_struct" = "Non-Structural", "Struct" = "Structural")) + 
  scale_fill_manual(
    name = 'Result',
    labels = c('0' = 'Negative', '1' = 'Positive'), 
    values = c('0' = '#7570b3', '1' = '#d99f02')
  ) +
  
  # Apply a clean theme
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(paste0(PWD, "struct_vs_nonstruct_distributions_af3_20250814.png"), width = 8, height = 6)
