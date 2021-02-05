library(mlbench)
library(ggplot2)

# carregando as amostras em espiral 
data <- mlbench.circle(100)

x <- data[['x']]
classes <- as.numeric(data[['classes']])

df <- data.frame(data)

x1 <- x[,1]
x2 <- x[,2]

# plotando as amostras
ggplot(df, aes(x1, x2)) + geom_point(color=classes)

source("treinaELM.R")
source("YELM.R")

p <- 30
#train <- treinaELM(x, y, p, 1)


seq_x = seq(-1,1, 0.01)
seq_y = seq(-1,1, 0.01)

for (i in 1:length(classes)){
  if (classes[i] == 2) {
    classes[i] = -1
  }
}

train <- treinaELM(x, classes, p=p, 1)
M<-matrix(0,nrow=length(seq_x),ncol=length(seq_y))
ci<-0

for(i in seq_x){
  ci<-ci+1
  cj<-0
  for (j in seq_y) {
    cj<-cj+1
    x<-cbind(i,j)
    M[ci,cj]<- YEML(x,train[[3]], train[[1]],1)
  }
}


plot(x1, 
     x2, 
     xlab = "x1", 
     ylab = "x2", 
     col=ifelse(classes==1,'red','black'),
     pch = 16)
par(new=TRUE)

contour(seq_x,seq_y,M,1,xlim=c(-1,1),ylim=c(-1,1), xaxt="n", yaxt="n")


