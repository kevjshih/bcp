function beta = learn_obj_logreg(roc)

counts = [roc.tp(1)>0; roc.tp(2:end)-roc.tp(1:end-1)];

labels = 2*double(counts>0) - 1;
feats = roc.conf(:);


beta = trainLogReg(feats(:)', labels(:)', 1e-4);
