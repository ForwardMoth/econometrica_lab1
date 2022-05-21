import pandas as pd, os,scipy.stats as sc, matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import numpy as np

"""Change columns name, 1 column - y, others - x_. Delete region column"""
def change_columns(data):
    columns = list(data.columns)
    i = 0
    for column in columns:
        if i == 0:
            data.rename(columns={column: "y"},inplace=True)
        else:
            data.rename(columns={column: "x" + str(i)},inplace=True)
        i += 1

"""Shows decribe statistic of data"""
def show_describe_stat(data):
    print(data.describe())

"""Create lineral regression model"""
def get_model(ind_data,data):
    formula = "{} ~ {}".format("y", "+".join(ind_data.columns.to_list()))
    # print(formula)
    model = ols(formula=formula,data=data).fit()
    return model

"""Get assessment of coefficients KLMMR and information about KLMMR"""
def get_model_info(model):
    print(model.summary())

"""Get residuals of lineral regression"""
def get_residuals(model):
    print("Residuals of lineral regression")
    print(model.resid)
    return model.resid

"""Check hypothesis"""
def check_normality_of_residuals(residuals):
    """calcutate paramers for plot"""
    residuals_mean, residuals_std = residuals.mean(),residuals.std()
    """setting size of histogram"""
    plt.figure(figsize=(9, 6))

    """building histogram of residuals"""
    histData = plt.hist(residuals) 
    
    """build red line, which fits to the normal distribution"""
    range_ = np.arange(min(residuals), max(residuals), 0.05) 
    normModel = sc.norm(residuals_mean, residuals_std) 
    coefY = len(residuals) * max([1, (max(histData[0]) // (normModel.pdf(residuals_mean) * len(residuals)))])
    plt.plot(range_, [normModel.pdf(x) * coefY for x in range_], color="r")

    """Make divisions on the abscissa axis (upper interval of the partition boundary)"""
    plt.xticks(histData[1])
    
    """Make Kolmogorov test about normal distribution"""
    KS_maxD, KS_PValue = sc.kstest(residuals, cdf="norm", args=(residuals_mean, residuals_std))

    """Creation of title"""
    plt.title("Histogram of the distribution of regression residues\n" +
            "Distribution: Normal\n" +
            "Kolmogorov-Smirnov test = {:.5}, p-value = {:.5}".format(KS_maxD, KS_PValue), fontsize=18)

    """Text axises"""
    plt.ylabel("No. of observations", fontsize=15) 
    plt.xlabel("Category (upper limits)", fontsize=15) 

    """Make grid and show histogram"""
    plt.grid()
    plt.show()

def check_multicollinearity(data):
    print(data.corr())

"""elimination of multicollinearity by including variables"""
def forward_selected(data, response):
    remaining = set(data.columns) 
    remaining.remove(response) 
    selected = [] 
    currentScore, bestNewScore = 0.0, 0.0 
    while remaining and currentScore == bestNewScore:
        scoresWithCandidates = [] 
        for candidate in remaining:
            score = ols(formula="{} ~ {}".format(response, " + ".join(selected + [candidate])), data=data).fit().rsquared_adj
            scoresWithCandidates.append((score, candidate))
        scoresWithCandidates.sort() 
        bestNewScore, bestCandidate = scoresWithCandidates.pop() 
        if currentScore < bestNewScore:
            remaining.remove(bestCandidate) 
            selected.append(bestCandidate) 
            currentScore = bestNewScore
    return ols(formula="{} ~ {}".format(response, " + ".join(sorted(selected))), data=data).fit()

"""elimination of multicollinearity by excluding variables"""
def backward_elimination(data, response):
    selected = set(data.columns) 
    selected.remove(response) 
    currentPValues = ols(formula="{} ~ {}".format(response, " + ".join(selected)), data=data).fit().pvalues
    for _ in range(len(data.columns)): 
        currentAdjR2 = -1.0 
        if (max(currentPValues) >= 0.05):
            for i in range(1, len(currentPValues)):
                candidateToRemove = currentPValues.axes[0][i] 
                newModel = ols(formula="{} ~ {}".format(response, " + ".join(selected - set([candidateToRemove]))), data=data).fit()
                newAdjR2 = newModel.rsquared_adj 
                if (currentAdjR2 < newAdjR2):
                    currentAdjR2, deletedVar, improvedPValues = newAdjR2, candidateToRemove, newModel.pvalues
        if (currentAdjR2 == -1.0):
            break
        selected.remove(deletedVar) 
        currentPValues = improvedPValues
    return ols(formula="{} ~ {}".format(response, " + ".join(sorted(selected))), data=data). fit()            

def main():
    print("Starting of execution lab 1")
    file_name = r'C:\Users\asus\projects\econometrica_lab1\данные.xlsx'
    sheet = 'Обработка'
    try:
        data = pd.read_excel(file_name,sheet_name=sheet)
        data.drop(data.columns[0],axis=1,inplace=True)
        change_columns(data)
        independed_variables = data.drop(columns='y')

        show_describe_stat(independed_variables)

        model = get_model(independed_variables,data)
        get_model_info(model)
        residuals = get_residuals(model)
        
        check_normality_of_residuals(residuals)

        check_multicollinearity(data)

        forward_model = forward_selected(data, "y")
        get_model_info(forward_model)

        backward_model = backward_elimination(data, "y")
        get_model_info(backward_model)
    except FileNotFoundError:
        print("File didn't found")

if __name__ == '__main__':
    main()