
#---------------------------------##
## Educational Data Mining with R ##
##    Presented by Okan Bulut     ##
##          June 25, 2019         ##
#---------------------------------##


# Packages to be used
packages <- c("data.table", "tidyverse","e1071", "caret", "mlr",
              "modelr", "randomForest", "rpart", "rpart.plot", 
              "GGally", "ggExtra")

install.packages(packages)

# Activate all packages
library("data.table")
library("tidyverse")
library("e1071")
library("caret")
library("mlr")
library("modelr")
library("randomForest")
library("rpart") 
library("rpart.plot") 
library("GGally")
library("ggExtra")

# fread function from data.table
pisa <- fread("pisa_turkey.csv", na.strings = "")
class(pisa)

# Base R function for csv files (DON'T RUN)
pisa <- read.csv("pisa_turkey.csv", header = TRUE)
class(pisa)

# See variable names
names(pisa)

# Preview the data
head(pisa)

# Dimensions
dim(pisa)

# Structure
str(pisa)

# Define new variables
pisa <- mutate(pisa,
               
               # Reorder the levels of grade 
               grade = factor(grade, 
                              levels = c("Grade 7", "Grade 8", "Grade 9", "Grade 10",
                                         "Grade 11", "Grade 12", "Grade 13", 
                                         "Ungraded")),
               
               # Define a numerical grade variable
               grade1 = (as.numeric(sapply(grade, function(x) {
                 if(x=="Grade 7") "7"
                 else if (x=="Grade 8") "8"
                 else if (x=="Grade 9") "9"
                 else if (x=="Grade 10") "10"
                 else if (x=="Grade 11") "11"
                 else if (x=="Grade 12") "12"
                 else if (x=="Grade 13") NA_character_
                 else if (x=="Ungraded") NA_character_}))),
               
               # Total learning time as hours
               learning = round(TMINS/60, 0),
               
               # Science performance based on OECD average
               science_oecd = as.factor(ifelse(science >= 493, "High", "Low")),
               
               # Science performance based on Turkey's average
               science_tr = as.factor(ifelse(science >= 422, "High", "Low")))


# By grade - summary
pisa %>%
  group_by(grade) %>%
  summarise(Count = n(),
            science = mean(science, na.rm = TRUE),
            computer = mean(computer, na.rm = TRUE),
            software = mean(software, na.rm = TRUE),
            internet = mean(internet, na.rm = TRUE),
            own.room = mean(own.room, na.rm = TRUE)
  )


# By grade and gender - summary
pisa %>%
  group_by(grade, gender) %>%
  summarise(Count = n(),
            science = mean(science, na.rm = TRUE),
            computer = mean(computer, na.rm = TRUE),
            software = mean(software, na.rm = TRUE),
            internet = mean(internet, na.rm = TRUE),
            own.room = mean(own.room, na.rm = TRUE)
  )


# Data visualizations

ggplot(data = pisa, 
       mapping = aes(x = grade, y = science)) +
  geom_boxplot() +
  labs(x=NULL, y="Science Scores") +
  theme_bw()


ggplot(data = pisa, 
       mapping = aes(x = grade, y = science)) +
  geom_boxplot() +
  labs(x=NULL, y="Science Scores") +
  geom_hline(yintercept = 493, linetype="dashed", color = "red", size = 1) +
  geom_hline(yintercept = 422, linetype="dashed", color = "blue", size = 1) +
  theme_bw()


ggplot(data = pisa, 
       mapping = aes(x = grade, y = science, fill = gender)) +
  geom_boxplot() +
  labs(x=NULL, y="Science Scores") +
  geom_hline(yintercept = 493, linetype="dashed", color = "red", size = 1) +
  theme_bw()


ggpairs(data = pisa,
        mapping = aes(color = gender),
        columns = c("reading", "science", "math"),
        upper = list(continuous = wrap("cor", size = 4.5))
)


p1 <- ggplot(data = pisa,
             mapping = aes(x = learning, y = science)) +
  geom_point() +
  geom_smooth(method = "loess") +
  labs(x = "Weekly Learning Time", y = "Science Scores") +
  theme_bw()

# Replace "histogram" with "boxplot" or "density" for other types
ggMarginal(p1, type = "histogram")


ggcorr(data = pisa[,c("science", "reading", "computer", "own.room", "quiet.study",
                      "ESCS", "SCIEEFF", "BELONG", "ANXTEST", "COOPERATE", "learning", 
                      "EMOSUPS", "grade1")],
       method = c("pairwise.complete.obs", "pearson"),
       label = TRUE, label_size = 4)


# Set the seed before splitting the data
set.seed(442019)

# We need to remove missing cases
pisa_nm <- na.omit(pisa)

# Split the data into training and test
index <- createDataPartition(pisa_nm$science_tr, p = 0.7, list = FALSE)
train <- pisa_nm[index, ]
test  <- pisa_nm[-index, ]

nrow(train)
nrow(test)

#### Decision trees ----

dt_fit1 <- rpart(formula = science_tr ~ grade1 + computer + own.room + ESCS + 
                   EMOSUPS + COOPERATE,
                 data = train,
                 method = "class", 
                 control = rpart.control(minsplit = 20, 
                                         cp = 0, 
                                         xval = 0),
                 parms = list(split = "gini"))

summary(dt_fit1)


dt_fit2 <- rpart(formula = science_tr ~ grade1 + computer + own.room + ESCS + 
                   EMOSUPS + COOPERATE,
                 data = train,
                 method = "class", 
                 control = rpart.control(minsplit = 20, 
                                         cp = 0.006, 
                                         xval = 0),
                 parms = list(split = "gini"))

rpart.plot(dt_fit2)
rpart.plot(dt_fit2, extra = 8, box.palette = "RdBu", shadow.col = "gray")

printcp(dt_fit2)
summary(dt_fit2)
varImp(dt_fit2)


dt_pred <- predict(dt_fit2, test) %>%
  as.data.frame()

head(dt_pred)

dt_pred <- mutate(dt_pred,
                  science_tr = as.factor(ifelse(High >= 0.5, "High", "Low"))) %>%
  select(science_tr)

confusionMatrix(dt_pred$science_tr, test$science_tr)


dt_fit3 <- rpart(formula = science_tr ~ grade1 + computer + own.room + ESCS + 
                   EMOSUPS + COOPERATE,
                 data = train,
                 method = "class", 
                 control = rpart.control(minsplit = 20,
                                         cp = 0,
                                         xval = 10),
                 parms = list(split = "gini"))

printcp(dt_fit3)
plotcp(dt_fit3)


### Random forest ----

rf_fit1 <- randomForest(formula = science_tr ~ grade1 + computer + own.room + ESCS + 
                          EMOSUPS + COOPERATE,
                        data = train,
                        importance = TRUE, 
                        ntree = 1000)

print(rf_fit1)
plot(rf_fit1)


rf_fit2 <- randomForest(formula = science_tr ~ grade1 + computer + own.room + ESCS + 
                          EMOSUPS + COOPERATE,
                        data = train,
                        importance = TRUE, 
                        ntree = 50)

print(rf_fit2)
sum(diag(rf_fit2$confusion)) / nrow(train)


importance(rf_fit2) %>%
  as.data.frame() %>%
  mutate(Predictors = row.names(.)) %>%
  arrange(desc(MeanDecreaseGini))

varImpPlot(rf_fit2, 
           main = "Importance of Variables for Science Performance")

rf_pred <- predict(rf_fit2, test) %>%
  as.data.frame() %>%
  mutate(science_tr = as.factor(`.`)) %>%
  select(science_tr)

confusionMatrix(rf_pred$science_tr, test$science_tr)


### Support vector machines ----

svc_fit <- svm(formula = science_tr ~ reading + math + grade1 + computer + own.room +
                 ESCS + EMOSUPS + COOPERATE, 
               data = train, 
               kernel = "linear")

summary(svc_fit)
plot(svc_fit, data = train, reading ~ math)


set.seed(1)
ran_obs <- sample(1:nrow(train), 500)
plot(svc_fit, data = train[ran_obs, ], reading ~ math)


tune_svc <- tune(svm, 
                 science_tr ~ reading + math + grade1 + computer + own.room +
                   ESCS + EMOSUPS + COOPERATE, 
                 data = train,
                 kernel="linear",
                 ranges = list(cost = c(.01, .1, 1, 5, 10)))

summary(tune_svc)
best_svc <- tune_svc$best.model
summary(best_svc)


#' Evaluates a classifier (e.g. SVM, logistic regression)
#' @param tab a confusion matrix
eval_classifier <- function(tab, print = F){
  n <- sum(tab)
  TN <- tab[2,2]
  FP <- tab[2,1]
  FN <- tab[1,2]
  TP <- tab[1,1]
  classify.rate <- (TP + TN) / n
  TP.rate <- TP / (TP + FN)
  TN.rate <- TN / (TN + FP)
  object <- data.frame(accuracy = classify.rate,
                       sensitivity = TP.rate,
                       specificity = TN.rate)
  return(object)
}

# to create a confusion matrix this order is important!
# observed values first and predict values second!
svc_train <- table(train$science_tr, predict(best_svc)) 
eval_classifier(svc_train)

svc_test <- table(test$science_tr, predict(best_svc, newdata = test))
eval_classifier(svc_test)


# Comparison to logistic regression

lr_fit <- glm(science_tr ~ reading + math + grade1 + computer + own.room + ESCS + 
                EMOSUPS + COOPERATE, 
              data = train, 
              family = "binomial")

coef(lr_fit)

lr_train <- table(train$science_tr, 
                  round(predict(lr_fit, type = "response")))
lr_train
eval_classifier(lr_train)


lr_test <- table(test$science_tr,
                 round(predict(lr_fit, newdata = test, type = "response")))
lr_test
eval_classifier(lr_test)
















