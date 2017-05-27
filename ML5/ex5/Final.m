function[f] = Final(X,y,X_test,ytest,lambda)


theta = trainLinearReg(X, y, lambda);
J = linearRegCostFunction(X_test, ytest, theta, 0);
f = J;