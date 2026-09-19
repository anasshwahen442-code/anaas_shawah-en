%% COPD Exacerbation Prediction — MATLAB Pilot Implementation
% Anas Shawah'en
% Basic MATLAB implementation of the clinical-only logistic regression
% baseline, replicating the core methodology of the Python pipeline
% (see: github.com/anasshwahen442-code/anaas_shawah-en)
%
% NOTE: This is a pilot on SYNTHETIC data (n=50). Not a clinical finding.
% Uses only the clinical/patient-reported columns from
% copd_multimodal_dataset.csv (audio MFCC columns are ignored here,
% matching the Python "clinical only" model configuration).

clear; clc; close all;

%% 1. Load the dataset
data = readtable('copd_multimodal_dataset.csv');

% Clinical-only predictors (matches the 9 features used in the Python
% "clinical only" models: FEV1, CRP, eosinophils, CAT, mMRC, adherence,
% activity, ER visits — plus Age as a baseline covariate)
predictorNames = {'Age', 'FEV1_pct_predicted', 'CRP_mg_L', ...
    'Eosinophils_cells_uL', 'CAT_Score', 'mMRC_Dyspnea', ...
    'Medication_Adherence_pct', 'Activity_Steps_per_day', ...
    'ER_Visits_Last_6mo'};
outcomeName = 'Exacerbation_90d';

X = data(:, predictorNames);
y = data.(outcomeName);

%% 2. Set up 5-fold cross-validation (same principle as the Python pipeline)
rng(42); % fixed seed for reproducibility, matches Python's random_state=42
cv = cvpartition(y, 'KFold', 5);

foldAUC = zeros(cv.NumTestSets, 1);

figure; hold on;
colors = lines(cv.NumTestSets);

for i = 1:cv.NumTestSets
    trainIdx = training(cv, i);
    testIdx  = test(cv, i);

    Xtrain = X(trainIdx, :);
    ytrain = y(trainIdx);
    Xtest  = X(testIdx, :);
    ytest  = y(testIdx);

    % Fit logistic regression (fitglm = MATLAB's equivalent of sklearn's LogisticRegression)
    mdl = fitglm(Xtrain, ytrain, 'Distribution', 'binomial', 'Link', 'logit');

    % Predict probabilities on the held-out fold
    scores = predict(mdl, Xtest);

    % Compute ROC curve and AUC for this fold
    [Xroc, Yroc, ~, AUC] = perfcurve(ytest, scores, 1);
    foldAUC(i) = AUC;

    plot(Xroc, Yroc, 'Color', colors(i,:), 'DisplayName', sprintf('Fold %d (AUC=%.3f)', i, AUC));
end

%% 3. Report results (mirrors the Python model_comparison.py output style)
plot([0 1], [0 1], 'k--', 'DisplayName', 'Chance');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('COPD Clinical-Only Logistic Regression — 5-Fold ROC (MATLAB, synthetic pilot n=50)');
legend('Location', 'southeast');
grid on;

fprintf('\n--- MATLAB Pilot Results (clinical-only, synthetic n=50) ---\n');
fprintf('Fold AUCs: %s\n', mat2str(round(foldAUC, 3)));
fprintf('Mean AUC : %.3f\n', mean(foldAUC));
fprintf('Std  AUC : %.3f\n', std(foldAUC));

% Save the figure for the repository
saveas(gcf, 'roc_curve_matlab.png');

fprintf('\nFigure saved as roc_curve_matlab.png\n');