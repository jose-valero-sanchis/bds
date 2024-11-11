
BackTrackingLocal <- function(trellisBT, decisionf=max)
{
	trellis <- trellisBT$trellis
	backi <- trellisBT$backi
	backj <- trellisBT$backj
	
	#busqueda de la posicion con el valor mayor o menor
	pos <- which(trellis==decisionf(trellis), arr.ind=TRUE)[1,]
	
	#print(pos)
	while (pos[1]!=1 & pos[2]!=1)
	{
		i <- pos[1]
		j <- pos[2]

		inew <- backi[i,j]
		jnew <- backj[i,j]
		cat(i-1,",",j-1," <- \t", inew-1,",",jnew-1,":\t",sep="")
		if(inew==i-1 & jnew==j-1)
		{
			cat(trellis[i,j],"MoS\t", rownames(trellis)[i],colnames(trellis)[j],"\n")
		}
		else if(inew==i & jnew==j-1) #insercion de v[j]
		{
			cat(trellis[i,j],"I\t", "-",colnames(trellis)[j],"\n")
		}
		else if(inew==i-1 & jnew==j) #borrado de u[i]
		{
			cat(trellis[i,j],"B\t", rownames(trellis)[i],"-","\n")
		}
		else if(inew==1 & jnew==1)
		{
			cat(trellis[i,j],"Ini\t","\n")

		}
		posNew <- c(inew,jnew)
		pos <- posNew
	}
}

########################################################################################



#S: Sequence 1
#R: Sequence 2
#delta: scoring matrix or function 
#decisionf={max,min, ...}, if delta is a distance matrix, then min; else max
#Peor: the worst value for scoring
SmithWaterman <- function(S,R,delta, decisionf=max,Peor=-Inf)
{
  u <- strsplit(S," ")[[1]]
  m <- length(u)

  v <- strsplit(R," ")[[1]]
	n <- length(v)

	trellis <- matrix(rep(0,m*n),nrow=m,ncol=n, 
		dimnames=list(u,v))

	backi <- matrix(rep(0,m*n),nrow=m,ncol=n, 
		dimnames=list(u,v))
	backj <- matrix(rep(0,m*n),nrow=m,ncol=n, 
		dimnames=list(u,v))

	costes <- c(Peor,Peor,Peor,0)
	names(costes) <- c("MoS", "I", "B","Ini")

	#borrado de u[i]
	for (i in 2:m)
	{
		costes["B"] <- trellis[i-1,1]+delta(u[i],"-")

		trellis[i,1] <- decisionf(costes)

		operacion <- names(which(costes==trellis[i,1])[1])
		if (operacion=="B")
			{backi[i,1] <- i-1; backj[i,1] <- 1;}
		else if (operacion=="Ini")
			{backi[i,1] <- 1; backj[i,1] <- 1;}
		else warning("Not recognized operation in Trellis")

	}

	costes <- c(Peor,Peor,Peor,0)
	names(costes) <- c("MoS", "I", "B","Ini")

	for (j in 2:n)
	{
		costes["I"] <- trellis[1,j-1]+delta("-",v[j])
		trellis[1,j] <- decisionf(costes)

		operacion <- names(which(costes==trellis[1,j])[1])
		if (operacion=="B")
			{backi[1,j] <- 1; backj[1,j] <- j-1;}
		else if (operacion=="Ini")
			{backi[1,j] <- 1; backj[1,j] <- 1;}
		else warning("Not recognized operation in Trellis")
	}

	costes <- c(Peor,Peor,Peor,0)
	names(costes) <- c("MoS", "I", "B","Ini")

	#General loop
	for (i in 2:m)
	{
		for (j in 2:n)
		{
		  #deletion of u[i]
			costes["B"] <-   #TODO EXERCICE 1
			
			#insertion of v[j]
			costes["I"] <-   #TODO EXERCICE 1

			#sustitution or match u[i]==v[j]
			costes["MoS"] <-   #TODO EXERCICE 1
			
			#cost of the operation
			trellis[i,j] <- decisionf(costes)
			
			#preparing the backtracking
			operacion <- names(which(costes==trellis[i,j])[1])
			
			if (operacion=="B")
				{backi[i,j] <- i-1; backj[i,j] <- j;}
			else if (operacion=="I")
				{backi[i,j] <- i; backj[i,j] <- j-1;}
			else if (operacion=="MoS")
				{backi[i,j] <- i-1; backj[i,j] <- j-1;}
			else if (operacion=="Ini")
				{backi[i,j] <- 1; backj[i,j] <- 1;}
			else warning("Not recognized operation in Trellis")

		} #for j
	} #for i

	result <- list()
	result$trellis <- trellis
	result$backi <- backi
	result$backj <- backj
	
	print(result$trellis)
	BackTrackingLocal(result) 
	return(result)
} #fin SW

############## definition of delta funcntion ###################

cholesterol_level <- function(CHO_TOTAL)
{
  #return "high", "borderline" or "normal" depending the level of total cholesterol according to the next guideline:
  #Total cholesterol levels less than 200 milligrams per deciliter (mg/dL) are considered desirable for adults. A reading between 200 and 239 mg/dL is considered borderline high, and a reading of 240 mg/dL and above is considered high.
  
  #TODO EXERCICE 2
  CHO_TOTAL_normal <- (CHO_TOTAL < 200.0)
  CHO_TOTAL_borderline <- (CHO_TOTAL >= 200.0) & (CHO_TOTAL < 240)
  CHO_TOTAL_high <-(CHO_TOTAL >= 240)
  if (CHO_TOTAL_high) return(#TODO EXERCISE 2)
  if (CHO_TOTAL_borderline) return(#TODO EXERCISE 2)
  if (CHO_TOTAL_normal) return(#TODO EXERCISE 2)
  
}

blood_preasure_level <- function(BP_systolic,BP_diastolic)
{
  #return "high", "prehypertension" or "normal", depending the levels of systolic and diastolic blood preasures according 
  #to the next guideline:
  # Normal	systolic: less than 120 mm Hg diastolic: less than 80 mm Hg
  # At Risk (prehypertension)	systolic: 120-139 mm Hg diastolic: 80-89 mm Hg
  # High Blood Pressure (hypertension)	systolic: 140 mm Hg or higher diastolic: 90 mm Hg or higher

  #TODO EXERCICE 3
  
  BP_normal <- (BP_systolic < 120.0) & (BP_diastolic < 80.0)
  BP_prehypertension <- (BP_systolic >= 120.0 & BP_systolic < 140.0 ) | (BP_diastolic >= 80.0 & BP_diastolic < 90.0 )
  BP_high <- (BP_systolic >= 140.0) | (BP_diastolic >= 90.0)
  if (BP_high) return(#TODO EXERCISE 3)
  if (BP_prehypertension) return(#TODO EXERCISE 3)
  if (BP_normal) return(#TODO EXERCISE 3)
  
}

#delta matrix for cholecterol levels
match <- 1
jump1 <- -1
jump2 <- -2
gap <- -1 #insert/delete 
delta_CHO_TOTAL <- matrix(c(match,jump1,jump2,gap,
                  jump1,match,jump1,gap,
                  jump2,jump1,match,gap,
                  gap,gap,gap,Inf), 
                dimnames=list(c("normal","borderline","high","-"),
                              c("normal","borderline","high","-")),
                nrow=4,ncol=4,byrow=TRUE)

#delta matrix for blood preasure levels
match <- 1
jump1 <- -1
jump2 <- -2
gap <- -1 #insert/delete 
delta_BLOOD_PREASURE <- matrix(c(match,jump1,jump2,gap,
                            jump1,match,jump1,gap,
                            jump2,jump1,match,gap,
                            gap,gap,gap,Inf), 
                          dimnames=list(c("normal","prehypertension","high","-"),
                                        c("normal","prehypertension","high","-")),
                          nrow=4,ncol=4,byrow=TRUE)

delta <- function(ui,vj)
{
  ui_elements <- strsplit(ui,"\\+")[[1]]
  eli <- ui_elements[1]
  if (eli != "-"){
    ui_CHO_TOTAL <- as.numeric(eli)
    ui_BP_systolic <- as.numeric(ui_elements[2]) 
    ui_BP_diastolic <- as.numeric(ui_elements[3])
    ui_cholesterol_level <- cholesterol_level(ui_CHO_TOTAL)
    ui_blood_preasure_level <- blood_preasure_level(ui_BP_systolic,ui_BP_diastolic)
  }
  else{
    ui_cholesterol_level <- "-"
    ui_blood_preasure_level <- "-"
  }
  
  vj_elements <- strsplit(vj,"\\+")[[1]]
  elj <- vj_elements[1]
  if (elj != "-"){
    vj_CHO_TOTAL <- as.numeric(elj)
    vj_BP_systolic <- as.numeric(vj_elements[2])
    vj_BP_diastolic <- as.numeric(vj_elements[3])
    vj_cholesterol_level <- cholesterol_level(vj_CHO_TOTAL)
    vj_blood_preasure_level <- blood_preasure_level(vj_BP_systolic,vj_BP_diastolic)
  }
  else{
    vj_cholesterol_level <- "-"
    vj_blood_preasure_level <- "-"
  }
  # return the score of the delta function by adding the delta functions for cholesterol and blood preasure levels

  return(#TODO EXERCISE 4)
  
}

# TEST THE ALGORITHM
# TODO EXERCICE 5: prepare three new clinical pathways with different conditions to test the algorithm

seq1 <- "80.0+110.0+70.0 80.0+110.0+75.0 80.0+110.0+70.0 80.0+110.0+70.0 80.0+110.0+70.0 80.0+110.0+70.0" #sin problemas de cardiopatia
seq2 <- "80.0+110.0+70.0 80.0+110.0+75.0 90.0+150.0+95.0 90.0+145.0+95.0 90.0+145.0+95.0 90.0+145.0+95.0" #con problemas sistole-diastole a partir de 4a visita
seq3 <- "80.0+110.0+70.0 80.0+110.0+75.0 80.0+110.0+75.0 90.0+150.0+95.0 90.0+145.0+95.0 90.0+145.0+95.0" #con problemas de sistole-diastole a partir de 5a visita

print(seq1)
print(seq1)
t11sw <- SmithWaterman(seq1,seq1,delta)

print(seq1)
print(seq2)
t12sw <- SmithWaterman(seq1,seq2,delta)

print(seq1)
print(seq3)
t13sw <- SmithWaterman(seq1,seq3,delta)

print(seq2)
print(seq3)
t23sw <- SmithWaterman(seq2,seq3,delta)

#seq 4 <- TODO EXERCICE 5
#seq 5 <- TODO EXERCICE 5
#seq 6 <- TODO EXERCICE 5

# print(seq4) #TODO EXERCICE 5
# print(seq5) #TODO EXERCICE 5
# t45sw <- SmithWaterman(seq4,seq5,delta) #TODO EXERCICE 5

# print(seq4) #TODO EXERCICE 5
# print(seq6) #TODO EXERCICE 5
# t46sw <- SmithWaterman(seq4,seq6,delta) #TODO EXERCICE 5

# print(seq5) #TODO EXERCICE 5
# print(seq6) #TODO EXERCICE 5
# t56sw <- SmithWaterman(seq5,seq6,delta) #TODO EXERCICE 5
