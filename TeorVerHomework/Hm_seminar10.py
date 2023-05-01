
import numpy as np
import scipy.stats as stats

alpha = 0.05

# даны значения роста в трех группах случайно выбранных спортсменов
football = np.array([173, 175, 180, 178, 177, 185, 183, 182])
hockey = np.array([177, 179, 180, 188, 177, 172, 171, 184, 180])
barbell = np.array([172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170])

# проверка на нормальность 
print(stats.shapiro(football)) # ShapiroResult(statistic=0.9775082468986511, pvalue=0.9495404362678528) 
print(stats.shapiro(hockey))   # ShapiroResult(statistic=0.9579196572303772, pvalue=0.7763139009475708)
print(stats.shapiro(barbell))  # ShapiroResult(statistic=0.9386808276176453, pvalue=0.5051165223121643) 
# pvalue больше alpha, что указывает на нормально распределение

# проверка на однородность дисперсий (количество значений в выборках различны)
print(stats.bartlett(football, hockey, barbell)) # BartlettResult(statistic=0.4640521043406442, pvalue=0.7929254656083131)
# pvalue больше alpha, что указывает на однородность дисперсий

k = 3
n = len(football) + len(hockey) + len(barbell) # n = 28

f_mean = np.mean(football)  # f_mean = 179.125
h_mean = np.mean(hockey)    # h_mean = 178.66666666666666
b_mean = np.mean(barbell)   # b_mean = 172.72727272727272

total = np.hstack([football, hockey, barbell])  
# [173 175 180 178 177 185 183 182 177 179 180 188 177 172 171 184 180 172 173 169 177 166 180 178 177 172 166 170]
t_mean = np.mean(total)    # t_mean = 176.46428571428572

# Сумма квадратов отклонений наблюдений от общего среднего
S_total = np.sum((total - t_mean)**2)   # S_total = 830.9642857142854

# Сумма квадратов отклонений средних групповых значений от общего среднего
S_fact = np.sum((f_mean - t_mean)**2)*len(football) + np.sum((h_mean - t_mean)**2)*len(hockey) + np.sum((b_mean - t_mean)**2)*len(barbell)
# S_fact = 253.9074675324678

# Остаточная сумма квадратов отклонений
S_ost = np.sum((football - f_mean)**2) + np.sum((hockey - h_mean)**2) + np.sum((barbell - b_mean)**2)
# S_ost = 577.0568181818182

# Рассчитаем дисперсии
D_fact = S_fact / (k-1)    # D_fact = 126.9537337662339
D_ost = S_ost / (n-k)      # D_ost = 23.08227272727273

# Наблюдаемый критерий Фишера
F_n = D_fact / D_ost       # F_n = 5.500053450812598
print("Наблюдаемый критерий Фишера =", F_n)

# Функция дисперсионного анализа
f = stats.f_oneway(football, hockey, barbell)
print(f)

# pvalue(0.01) < alpha(0.05) - в результате анализа найдены статистически значимые различия,
# т.е. верна гипотеза H1

# Критерий Фишера по таблице (на уровне значимости 0.05, степени свободы df1 = k - 1 = 2, df2 = n -k = 25)
# равен 3.38

# Критерий Фишера наблюдаемый(5.5) больше критерия Фишера табличного (3.38), что значит, что 
# в результате анализа найдены статистически значимые различия, т.е. верна гипотеза H1