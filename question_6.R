####Question 6, Part (i)
q6.data=read.csv("hw1q6_data.csv")
attach(q6.data)
View(q6.data)
nrow(q6.data[which(Class=="1"),])
nrow(q6.data[which(Class=="0"),])
####
####missing_rate; question(ii)
missing_rate_Glucose=nrow(q6.data[which(Glucose==0),])/nrow(q6.data)
missing_rate_BP=nrow(q6.data[which(BloodPressure==0),])/nrow(q6.data)
missing_rate_STh=nrow(q6.data[which(SkinThickness==0),])/nrow(q6.data)
missing_rate_BMI=nrow(q6.data[which(BMI==0),])/nrow(q6.data)
missing_rate_BP=nrow(q6.data[which(DiabetesPedigreeFunction==0),])/nrow(q6.data)
missing_rate_Age=nrow(q6.data[which(Age==0),])/nrow(q6.data)

###
#new_data=rm(q6.data[which(Glucose==0),])
df <- data.frame(q6.data)
result <- data.frame(newdata[which(Glucose=="1")])
# df1= data.frame(df$Glucose,df$BloodPressure,df$SkinThickness,df$BMI,df$DiabetesPedigreeFunction)
# df1[df1==0] <- NA
df
df[df==0] <- NA
df
df1=df[complete.cases(df[,1:2]),]
df1
df2=df1[complete.cases(df1[,3:4]),]
df2
df3=df2[complete.cases(df2[,5:6]),]
df3[is.na(df3)]<-0
df3
clean_set=df3
nrow(df3[which(df3$Class==1),])
nrow(df3[which(df3$Class==0),])
summary(clean_set$Glucose)
summary(clean_set$BloodPressure)
summary(clean_set$SkinThickness)
summary(clean_set$BMI)
summary(clean_set$DiabetesPedigreeFunction)
summary(clean_set$Age)
par(mfrow=c(1,2))
hist(clean_set$BloodPressure,breaks=10,col="light blue",xlab="Blood Pressure",ylab="Frequency",main="Blood pressure")
hist(clean_set$DiabetesPedigreeFunction,breaks=10,col="light blue",xlab="Diabetes",ylab="Frequency",main="Diabetes")
qqnorm(clean_set$BloodPressure,main="Blood pressure,")
qqnorm(clean_set$DiabetesPedigreeFunction,main="Diabetes")


# missing_rate_Glucose=nrow(q6.data[which(Glucose==0),])/nrow(q6.data)
##R doesn't comment in blocks, clt+shift+c for each line


