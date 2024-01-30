library(ordinalClust)

data("dataqol")

set.seed(1)

nbSEM <- 150
nbSEMburn <- 100
nbindmini <- 1
init <- "randomBurnin"

percentRandomB <- c(50, 50)

x <- as.matrix(dataqol;.classif[, 2:29])
v <- as.vector(dataqol.classif$death)

nb.sample <- ceiling(nrow(x) *7/10)
sample.train <- sample(1:nrow(x), nb.sample)

x.train <- x[sample.train, ]
x.validation <- x[-sample.train, ]


v.train <- v[sample.train, ]
v.validation <- v[-sample.train, ]