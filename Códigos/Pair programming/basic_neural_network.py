from numpy import random
import matplotlib.pyplot as plt

i = 1.5
y = 0.5
w = []
loss = []
historico_pesos = [[], [], [], [], [], [], [], []]
lr = 0.005

random.seed()

for _ in range(8):
    w.append(random.randint(-1, 1))

print(f'Pesos iniciados: {w}\n')

for epoch in range(1000):

    for pesos in range(8):
        historico_pesos[pesos].append(w[pesos])

    #Input --> 1ª hidden layer
    a0 = i * w[0]
    a1 = i * w[1]

    #1ª hidden layer --> 2ª hidden layer
    a2 = a0 * w[2] + a1 * w[3]
    a3 = a0 * w[4] + a1 * w[5]

    #2ª hidden layer --> Output
    a = a2 * w[6] + a3 * w[7]

    C = (a-y)**2        #Função custo
    dC_da = 2 * (a-y)   #Derivada da função custo

    da_dw7 = a3
    dC_dw7 = dC_da * da_dw7
    w[7] -= dC_dw7 * lr

    da_dw6 = a2
    dC_dw6 = dC_da * da_dw6
    w[6] -= dC_dw6 * lr

    da_da3 = w[7]
    da3_dw5 = a1
    dC_dw5 = dC_da * da_da3 * da3_dw5
    w[5] -= dC_dw5 * lr

    da3_dw4 = a0
    dC_dw4 = dC_da * da_da3 * da3_dw4
    w[4] -= dC_dw4 * lr

    da_da2 = w[6]
    da2_dw2 = a0
    dC_dw2 = dC_da * da_da2 * da2_dw2
    w[2] -= dC_dw2 * lr

    da2_dw3 = a1
    dC_dw3 = dC_da * da_da2 * da2_dw3
    w[3] -= dC_dw3 * lr

    da1_dw1 = i
    da3_da1 = w[5]
    da2_da1 = w[3]
    dC_dw1 = dC_da * da_da3 * da3_da1 * da1_dw1 + dC_da * da_da2 * da2_da1 * da1_dw1
    w[1] -= dC_dw1 * lr

    da0_dw0 = i
    da2_da0 = w[2]
    da3_da0 = w[4]
    dC_dw0 = dC_da * da_da2 * da2_da0 * da0_dw0 + dC_da * da_da3 * da3_da0 * da0_dw0
    w[0] -= dC_dw0 * lr

    loss.append(C)

    print(f'Loss: {C} - Epoch:{epoch + 1}')
    print(f'Valor de a: {a}\n')


plt.plot(range(1000), loss, label='Loss')
plt.plot(range(1000), historico_pesos[0], label='w0')
plt.plot(range(1000), historico_pesos[1], label='w1')
plt.plot(range(1000), historico_pesos[2], label='w2')
plt.plot(range(1000), historico_pesos[3], label='w3')
plt.plot(range(1000), historico_pesos[4], label='w4')
plt.plot(range(1000), historico_pesos[5], label='w5')
plt.plot(range(1000), historico_pesos[6], label='w6')
plt.plot(range(1000), historico_pesos[7], label='w7')
plt.legend()
plt.show()