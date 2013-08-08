function compare_curves(roc)

beta = learn_obj_prob(roc, 2);

figure;
plot(roc.r, roc.p, 'linewidth', 3);
hold on;
plot(roc.r, cumsum(sigmoid(roc.conf, beta))./(1:length(roc.conf))', 'r');

beta = learn_obj_prob2(roc, 2);
plot(roc.r, cumsum(sigmoid(roc.conf, beta))./(1:length(roc.conf))', 'g');

beta = learn_obj_logreg(roc);
plot(roc.r, cumsum(sigmoid(roc.conf, beta))./(1:length(roc.conf))', 'b');


legend('Real ROC',  'Histogram', 'Regressed', 'Logreg');
