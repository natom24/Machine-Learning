##################################
##
##   Analysis of Hemocyte Counts
##          Haulk Data
##
##           05/15/23
##################################

## Setwd
setwd("C:/Users/natom/Downloads/")

## Import data
Hemo.Counts  <- read.csv("hemocytes.csv",
  colClasses = c("numeric", "numeric", "factor", "factor", "factor"),
  header = TRUE)

## Look at the data
## quartz()
boxplot(counts ~ temp:treatment:inf, data = Hemo.Counts)

## One-way ANOVA - Uninfected Controls
OneWay.hemo.lm  <- lm(counts ~ temp,
           data = Hemo.Counts[Hemo.Counts$inf == "uninf" & Hemo.Counts$treatment == "control",])

## Check ANOVA
plot(OneWay.hemo.lm)

## Results ANOVA
anova(OneWay.hemo.lm)

## Two-way ANOVA - Infected Controls & Coevolved
TwoWay.hemo.lm  <- lm(counts ~ temp*treatment,
           data = Hemo.Counts[Hemo.Counts$inf == "inf",])

## Check ANOVA
plot(TwoWay.hemo.lm)

## Results ANOVA
anova(TwoWay.hemo.lm)

ThreeWay.hemo.lm <- lm(counts ~ temp*treatment*inf, data = Hemo.Counts)

anova(ThreeWay.hemo.lm)

## Make the uninfected temp graph
boxplot(counts ~ temp, data = Hemo.Counts[Hemo.Counts$inf == "uninf" & Hemo.Counts$treatment == "control",], boxwex = .25, outline = FALSE)

boxplot(counts ~ temp:treatment, data = Hemo.Counts[Hemo.Counts$inf == "inf",],xlab = "treatment", boxwex = .5, outline = FALSE)

