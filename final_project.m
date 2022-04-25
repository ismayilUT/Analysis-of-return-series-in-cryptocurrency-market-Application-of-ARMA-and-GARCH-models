clear;clc;close all

load mybtcdata.mat

pBTC=BTCUSD.AdjClose;
dBTC=BTCUSD.Date;
pETH=ETHUSD.AdjClose;
dETH=ETHUSD.Date;
pLTC=LTCUSD.AdjClose;
dLTC=LTCUSD.Date;

%let's see price changes of cryptocurrencies in graph format
subplot(3,1,1)
plot(dBTC,pBTC)
title('BTC prices')
ylabel Price
xlabel time
subplot(3,1,2)
plot(dETH,pETH)
title('ETH prices')
ylabel Price
xlabel time
subplot(3,1,3)
plot(dLTC,pLTC)
title('LTC prices')
ylabel Price
xlabel time
close

%Calculation of  returns
rBTC=100*diff(log(pBTC)); 
rETH=100*diff(log(pETH)); 
rLTC=100*diff(log(pLTC));

%sychnronizing our data: creation of  time series structures
tsbtc=timeseries(rBTC,datenum(dBTC(2:end)));
tseth=timeseries(rETH,datenum(dETH(2:end)));
tsltc=timeseries(rLTC,datenum(dLTC(2:end)));

%Let's sync. data
[tsbtc,tsltc]=synchronize(tsbtc,tsltc,'Intersection','Interval',1);
[tsbtc,tseth]=synchronize(tsbtc,tseth,'Intersection','Interval',1);
[tseth,tsltc]=synchronize(tseth,tsltc,'Intersection','Interval',1);

%Verification of the sync whether worked or not:
all(tsbtc.Time==tsltc.Time)
all(tsbtc.Time==tseth.Time)
all(tseth.Time==tsltc.Time)

%Recover the synchronized data 
rBTC=tsbtc.data; rETH=tseth.data; rLTC=tsltc.data;
date=tsbtc.time;
date=datetime(date,'ConvertFrom','datenum');

%are the price series stationary? Augmented Dickey Fuller test
%adftest: the null hypothesis states that the series is non-stationary
[hpADB,p]=adftest(pBTC);%the series is non-stationary
[hpADE,p]=adftest(pETH);%the series is non-stationary
[hpADL,p]=adftest(pLTC);%the series is non-stationary


%are the return series stationary? Augmented Dickey Fuller test(again)
[hrADB,p]=adftest(rBTC) %the return series is stationary
[hrADE,p]=adftest(rETH) %the return series is stationary
[hrADL,p]=adftest(rLTC) %the return series is stationary

% Checking whether the returns normally distributed or not?!
% With Jarque-Bera test
%Null hypothesis states that return series is normally distributed.
% Since we reject H0, the return series is not normally distributed.
[hrJBB,p]=jbtest(rBTC)%the return series is not normally distributed
[hrJBE,p]=jbtest(rETH)%the return series is not normally distributed
[hrJBL,p]=jbtest(rLTC)%the return series is not normally distributed
% Jarque-Bera test demostrates us that none of the return series is 
% normally distributed

%descriptive statistics
returnmat=[rBTC,rETH,rLTC];
descstat.mean=mean(returnmat);%mean
descstat.var=var(returnmat);%variance
descstat.skew=skewness(returnmat);%skewness
descstat.kur=kurtosis(returnmat);%kurtosis
descstat.med=median(returnmat);%median, quantile(returnmat,0.5)
descstat.q05=quantile(returnmat,0.05);%5th quantile (VaR: value at risk)
descstat.q95=quantile(returnmat,0.95);%95th quantile
descstat;

%Plotting returns
subplot(3,1,1)
plot(date,rBTC)
title('BTC Returns')
subplot(3,1,2)
plot(date,rETH)
title('ETH Returns')
subplot(3,1,3)
plot(date,rLTC)
title('LTC Returns')
close 

%Are the data vectors normally distributed? 
subplot(1,3,1)
qqplot(rBTC)
title('BTC returns')
subplot(1,3,2)
qqplot(rETH)
title('ETH returns')
subplot(1,3,3)
qqplot(rLTC)
title('LTC returns')
close
%QQ-plots of returns of Cryptocurrencies also proves that
% there is no normally distributed series 


indp=find(rETH>mean(rETH)+4*std(rETH))%indices of positive outliers
indn=find(rETH<mean(rETH)-5*std(rETH))%indices of negative outliers
% These codes have been replicated from your classes. But in the case
% of independent variables(BTC, LTC) we could not integrate outliers
% method to our model.

%Our dependent variable is: Ethereum returns
subplot(2,2,1)
ksdensity(rETH)%kernel density estimate of the distribution
title('Empirical distribution of ETH returns')
subplot(2,2,2)
plot(date,rETH)
title('ETH returns')
subplot(2,2,3)
boxplot(rETH)
title('Box Plot of ETH returns')
subplot(2,2,4)
hist(rETH,100)
title('Histogram of ETH returns')
close

%Bitcoin returns 
%autocorrelation and partial autocorrelations functions with MFE toolbox
subplot(1,2,1)
autocorr(rETH,20)
subplot(1,2,2)
parcorr(rETH,20)
close

% Ljung-Box test for detemining the serial autocorrelation
[hBTC,p]=lbqtest(rBTC,'Lag',20)
[hETH,p]=lbqtest(rETH,'Lag',20)
[hLTC,p]=lbqtest(rLTC,'Lag',20)
%From here we came to conclusion to choose ETH returns as dependent 
%variable which shows serial autocorrelation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Let's estimate models where Bitcoin and Litecoin returns affect the 
%Ethereum returns. We are taking into account t time for exogenous 
%varibales otherwise we would be deleted reverse causality from the model
%and model would show us inconsistency and bias coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                %%%X exogenous variables with 1 lag%%%

%ARMA(4,3)X(1)
%rETH_t=alpha+beta1*rETH_t-1+beta2*rETH_t-2+beta3*rETH_t-3+beta4*rETH_t-4+
%delta*rBTC_t+delta1*rBTC_t-1+kappa*rLTC_t+kappa1*rLTC_t-1+eps_t+
%theta1*eps_t-1+theta2*eps_t-2+theta3*eps_t-3
C=1;% Constant intercept 
p=1:4;%AR order
q=1:3;%MA order
Xm=[[0;rBTC(2:end)] [0;rBTC(1:end-1)] [0;rLTC(2:end)] [0;rLTC(1:end-1)]]
[parest_ARMAX43,LL_ARMAX43,ERR_ARMAX43,~,DIAG_ARMAX43,VCVR_ARMAX43]=...
    armaxfilter(rETH,C,p,q,Xm)
AIC_ARMA43X1=DIAG_ARMAX43.AIC;
[h43,p43]=lbqtest(ERR_ARMAX43, 'Lags',20)
se=sqrt(diag(VCVR_ARMAX43));
estres=[parest_ARMAX43'; se'; 2*(1-normcdf(abs(parest_ARMAX43./se)))']


%ARMA(4,4)X(1)
%rETH_t=alpha+beta1*rETH_t-1+beta2*rETH_t-2+beta3*rETH_t-3+beta4*rETH_t-4+
%delta*rBTC_t+delta1*rBTC_t-1+kappa*rLTC_t+kappa1*rLTC_t-1+eps_t+
%theta1*eps_t-1+theta2*eps_t-2+theta3*eps_t-3+theta4*eps_t-4
C=1;% Constant intercept 
p=1:4;%AR order
q=1:4;%MA order
[parest_ARMAX44,LL_ARMAX44,ERR_ARMAX44,~,DIAG_ARMAX44,VCVR_ARMAX44]=...
    armaxfilter(rETH,C,p,q,Xm)
AIC_ARMA44X1=DIAG_ARMAX44.AIC;
[h44,p44]=lbqtest(ERR_ARMAX44, 'Lags',20)
se=sqrt(diag(VCVR_ARMAX44));
estres=[parest_ARMAX44'; se'; 2*(1-normcdf(abs(parest_ARMAX44./se)))']
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                 %%%X exogenous variables with 3 lag%%%

%ARMAX(5,3) with 3 lags:
%rETH_t=alpha+beta1*rETH_t-1+beta2*rETH_t-2+beta3*rETH_t-3+beta4*rETH_t-4+
% +beta5*rETH_t-5+delta*rBTC_t+ +delta1*rBTC_t-1+delta2*rBTC_t-2+
% +delta3*rBTC_t-3+kappa*rLTC_t+kappa1*rLTC_t-1+kappa2*rLTC_t-2+
% +kappa3*rLTC_t-3+eps_t+theta1*eps_t-1+theta2*eps_t-2+theta3*eps_t-3
C=1;% Constant intercept 
p=1:5;%AR order
q=1:3;%MA order
Xm3=[rBTC(4:end) rBTC(3:end-1) rBTC(2:end-2) rBTC(1:end-3) rLTC(4:end) rLTC(3:end-1) rLTC(2:end-2) rLTC(1:end-3)] %rBTC(4:end) rLTC(4:end)
[parest_ARMA53X3,LL_ARMA53X3,ERR_ARMA53X3,~,DIAG_ARMA53X3,VCVR_ARMA53X3]=...
    armaxfilter(rETH(4:end),C,p,q,Xm3)
AIC_ARMA53X3=DIAG_ARMA53X3.AIC;
[h533,p533]=lbqtest(ERR_ARMA53X3,'Lags',20)
se=sqrt(diag(VCVR_ARMA53X3));
estres=[parest_ARMA53X3'; se'; 2*(1-normcdf(abs(parest_ARMA53X3./se)))']

%ARMAX(5,4) with 3 lags:
%rETH_t=alpha+beta1*rETH_t-1+beta2*rETH_t-2+beta3*rETH_t-3+beta4*rETH_t-4+
% +beta5*rETH_t-5+delta*rBTC_t+ +delta1*rBTC_t-1+delta2*rBTC_t-2+
% +delta3*rBTC_t-3+kappa*rLTC_t+kappa1*rLTC_t-1+kappa2*rLTC_t-2+
% +kappa3*rLTC_t-3+eps_t+theta1*eps_t-1+theta2*eps_tp2+theta3*eps_t-3+
% theta4*eps_t-4
C=1;% Constant intercept 
p=1:5;%AR order
q=1:4;%MA order
[parest_ARMA54X3,LL_ARMA54X3,ERR_ARMA54X3,~,DIAG_ARMA54X3,VCVR_ARMA54X3]=...
    armaxfilter(rETH(4:end),C,p,q,Xm3)
AIC_ARMA54X3=DIAG_ARMA54X3.AIC;
[h543,p543]=lbqtest(ERR_ARMA54X3,'Lags',20)
se=sqrt(diag(VCVR_ARMA54X3));
estres=[parest_ARMA54X3'; se'; 2*(1-normcdf(abs(parest_ARMA54X3./se)))']




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                 %%%X exogenous variables with 2 lag%%%


%ARMA(4,2)X(2)
%rETH_t=alpha+beta1*rETH_t-1+beta2*rETH_t-2+beta3*rETH_t-3+beta4*rETH_t-4+
%delta*rBTC_t++delta1*rBTC_t-1+delta2*rBTC_t-2++kappa*rLTC_t+kappa1*rLTC_t-1+
%kappa2*rLTC_t-2+eps_t+theta1*eps_t-1+theta2*eps_t-2
C=1;% Constant intercept 
p=1:4;%AR order
q=1:2;%MA order
Xm2=[rBTC(3:end) rBTC(2:end-1) rBTC(1:end-2) rLTC(3:end) rLTC(2:end-1) rLTC(1:end-2)]
[parest_ARMA42X2,LL_ARMA42X2,ERR_ARMA42X2,~,DIAG_ARMA42X2,VCVR_ARMA42X2]=...
    armaxfilter(rETH(3:end),C,p,q,Xm2)
se=sqrt(diag(VCVR_ARMA42X2));
estres_ARMA42X2=[parest_ARMA42X2';se'; 2*(1-normcdf(abs(parest_ARMA42X2./se)))']
AIC_ARMA42X2=DIAG_ARMA42X2.AIC;
[h422,p422]=lbqtest(ERR_ARMA42X2,'Lags',20)


%ARMA(4,4)X(2)
%rETH_t=alpha+beta1*rETH_t-1+beta2*rETH_t-2+beta3*rETH_t-3+beta4*rETH_t-4+
%delta*rBTC_t++delta1*rBTC_t-1+delta2*rBTC_t-2++kappa*rLTC_t+kappa1*rLTC_t-1+
%kappa2*rLTC_t-2+eps_t+theta1*eps_t-1+theta2*eps_t-2+theta3*eps_t-3+
%+theta4*eps_t-4
C=1;% Constant intercept 
p=1:4;%AR order
q=1:4;%MA order
Xm2=[rBTC(3:end) rBTC(2:end-1) rBTC(1:end-2) rLTC(3:end) rLTC(2:end-1) rLTC(1:end-2)]
[parest_ARMA44X2,LL_ARMA44X2,ERR_ARMA44X2,~,DIAG_ARMA44X2,VCVR_ARMA44X2]=...
    armaxfilter(rETH(3:end),C,p,q,Xm2)
se=sqrt(diag(VCVR_ARMA44X2));
estres_ARMA44X2=[parest_ARMA44X2';se'; 2*(1-normcdf(abs(parest_ARMA44X2./se)))']
AIC_ARMA44X2=DIAG_ARMA44X2.AIC;
[h442,p442]=lbqtest(ERR_ARMA44X2,'Lags',20)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%We estimated approximately 18 models and choose ARMA53X3 model which 
%removed serial autocorrelation from series and gave us best AIC(AKAIKE)
% information criterion(1.8914). For the further exploration we need to 
% establish volatility models and evaluate them.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                      %%%Volatility models%%%

%Ljung Box test for autocorrelation in returns (again)
[hETH,p]=lbqtest(rETH,'Lags',20)
%The null hypothesis that there is no autocorrelation in returns is
%rejected.
%Ljung Box test to see if residuals are serially correlated
[hERRE,pERRE]=lbqtest(ERR_ARMA53X3,'Lags',20)
%yes, the fit was successful, serial correlation is removed from the
%residuals.
res=ERR_ARMA53X3
T=length(res);%length of the standart errors has been obtained form
%ARMAX53X3 model.
%Let's look at how the squared residuals are:
plot(res.^2)

%Let's test for serial correlation in squared residuals
[hSERRE,pSERRE]=lbqtest(res.^2,'Lags',20)
%yes, there is serial correlation in squared residuals.

%ARCH test
[hERREarch,pERREarch]=archtest(res, 'Lags', 20)
% Null hypothesis states there is not heteroscedasticity and we do not need
% create and estimate volatility model. In this test we reject H0.

%Riskmetrics: filter
%For daily data: lambda=0.94
h=var(res);%to start the iterations, we will use the sample variance
lam=0.94;T=length(res);
%length of the standart errors has been obtained form ARMAX53X3 model.
for i=2:T; %T is data length
    h(i,1)=(1-lam)*res(i-1)^2+lam*h(i-1,1);
end
plot(abs(res))
hold on
plot(sqrt(h),'r','LineWidth',2)

%ARCH(1) model
p=1;%arch order
o=0;%asymmetry order
q=0;%garch order
errortype='NORMAL';%normal distribution
tarchtype=2;%model is defined in squares 
init=[0.1;0.4];
[parameters,LLarch, ht, VCVrob]=tarch(res, p, o, q, errortype, tarchtype, init);
se=sqrt(diag(VCVrob));%standard errors
pval=2*(1-normcdf(abs(parameters./se)));
[parameters';se';pval']
AICarch=aicbic(LLarch, length(parameters), length(rETH))

%GARCH(1) model
p=1;%arch order
o=0;%asymmetry order
q=1;%garch order
errortype='NORMAL';%normal distribution
tarchtype=2;%model is defined in squares 
init=[0.1;0.2;0.7];
[parameters,LLgarch, ht, VCVrob]=tarch(res, p, o, q, errortype, tarchtype, init);
se=sqrt(diag(VCVrob));%standard errors
pval=2*(1-normcdf(abs(parameters./se)));
[parameters';se';pval']
AICgarch=aicbic(LLgarch, length(parameters), length(rETH))

%GJR-GARCH(1) model
p=1;%arch order
o=1;%asymmetry order: GJR-GARCH
q=1;%garch order
errortype='NORMAL';%normal distribution
tarchtype=2;%model is defined in squares 
init=[0.1;0.2;0.05;0.7];
[parameters,LLgjrgarch, ht, VCVrob]=tarch(res, p, o, q, errortype, tarchtype, init);
se=sqrt(diag(VCVrob));%standard errors
pval=2*(1-normcdf(abs(parameters./se)));
[parameters';se';pval']
AICgjrgarch=aicbic(LLgjrgarch, length(parameters), length(rETH))

%NAGARCH(1) model
p=1;%arch order
q=1;%garch order
errortype='NORMAL';%normal distribution
modeltype='NAGARCH';
tarchtype=2;%model is defined in squares 
init=[0.1;0.2;0.05;0.7];
[parameters,LLnagarch, ht, VCVrob]=agarch(res, p, q, modeltype, errortype, init);
se=sqrt(diag(VCVrob));%standard errors
pval=2*(1-normcdf(abs(parameters./se)));
[parameters';se';pval']
AICnagarch=aicbic(LLnagarch, length(parameters), length(rETH))

%Among ARCH GARCH and GJRGARCH models  GARCH shows less significance(0.0469)
%with stronger theta1 parameter value(0.1056). Also GARCH model has smaller
%AIC(Akaike) information criterion which gives better model option.
%Between GARCH and NAGARCH(nonlinear) we can observe that theta1 parameter has
%stronger value with less significance in GARCH model. At last but not least,
%we compared AKAIKE information criterion between them and concluded that 
%the best volatility model is  GJRGARCH model for our residuals and analysis.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                   %%%Vector autoregressive model%%%

Cons=1;
Lags=1:2;
Het=1; %default: heteroscedastic errors
Uncorr=0; %default: serially correlated errors
[PARAMETERS,STDERR,TSTAT,PVAL,CONST,CONSTSTD,R2,ERRORS,S2,PARAMVEC,VEC]...
         = vectorar([rBTC rLTC rETH],Cons,Lags,Het,Uncorr)
pr=cell2mat(PARAMETERS)';
std=cell2mat(STDERR)';
pv=cell2mat(PVAL)';
[pr;std;pv]



%CCC-MVGARCH
P=1; O=0; Q=1;
gjrtype=1;
data=ERRORS;
[parestccc,loglikeccc,Htccc,VCVccc] = ccc_mvgarch(data,[],P,O,Q,gjrtype);
sterr=sqrt(diag(VCVccc));
[parestccc';sterr']



%DCC-MVGARCH
P=1; O=0; Q=1;
gjrtype=1;
data=ERRORS;
[parestdcc,loglikedcc,Htdcc,VCVdcc] = dcc(data,[],P,O,Q,gjrtype);
sterr=sqrt(diag(VCVdcc));
[parestdcc;sterr']


for i=1:T;
    Dt=sqrt(diag(Htdcc(:,:,i)));
    temp=pinv(diag(Dt))*Htdcc(:,:,i)*pinv(diag(Dt));%Dt^(-1)*Ht*Dt^(-1)
    R(i,1)=temp(2,1);
end

subplot(4,1,1)
plot(date(4:end),R)
title Correlations
subplot(4,1,2)
plot(date,rBTC)
title('BTC Returns')
subplot(4,1,3)
plot(date,rETH)
title('ETH Returns')
subplot(4,1,4)
plot(date,rLTC)
title('LTC Returns')























