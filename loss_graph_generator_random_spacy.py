import matplotlib.pyplot as plt
# left is d loss right is G loss
output = """
 1.5116262435913086, -0.6157708764076233   |
 0.3741908371448517, -5.670180797576904   |
 0.4171394109725952, -6.640296936035156    |
 0.41705381870269775, -6.607815265655518   | 
 0.43326666951179504, -6.6704020500183105  | 
 0.44407856464385986, -7.777468681335449   | 
 0.49280425906181335, -7.976546764373779   | 
 0.4386473298072815, -6.9292826652526855   | 
"""

steps = []
DLoss = []
Gloss = []
losses = output.split("|")
for step, loss in enumerate(losses[:-1]):
    steps.append(step*500)
    loss = loss.split(",")
    print(loss)
    DLoss.append(float(loss[0].strip()))
    Gloss.append(float(loss[1].strip()))

plt.figure(figsize=(10,5))

# Plotting Discriminator loss
plt.plot(steps, DLoss, label='D Loss')

# Plotting Generator loss
plt.plot(steps, Gloss, label='G Loss')

plt.title('Losses vs Steps BERT')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()

plt.show()


