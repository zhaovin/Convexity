### R code from vignette source 'knitr-intro.Rnw'

###################################################
### code chunk number 1: show-off
###################################################
rnorm(5)
df=data.frame(y=rnorm(100), x=1:100)
summary(lm(y~x, data=df))


