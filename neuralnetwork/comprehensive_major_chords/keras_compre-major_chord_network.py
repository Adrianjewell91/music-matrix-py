from keras.api.models import Sequential
from keras.api.layers import Dense
import numpy

# fix random seed for reproducibility
#this line makes sure that theta values are randomized to same value for every run.
#It is not a necessary line. - should ask the tutorial writer about it.
numpy.random.seed(88)

dataset = numpy.loadtxt("major-chords-compre-set-train.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:88]
Y = dataset[:,88]

from keras.api.utils import to_categorical
Y_train = to_categorical(Y)

# validate_dataset = numpy.loadtxt("major-chords-compre-set-validate.csv", delimiter=",")
validate_dataset = numpy.loadtxt("test.csv", delimiter=",")
X_validate = validate_dataset[:,0:88]
Y_validate = validate_dataset[:,88]

Y_train_validate = to_categorical(Y_validate)

#create model
model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(44, input_dim=88, activation='sigmoid'))
model.add(Dense(12, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model
model.fit(X,Y_train, epochs=100, batch_size=4)

scores = model.evaluate(X_validate,Y_train_validate, verbose=1)

print ("Evaluation of the network using the test version")

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Test against the sample chords, the model will guess correctly by outputting the largest probability against
# the correct pitch center.

# In the test, the correct pitch centers for the first four samples are:
# 10 (Bflat),
# 11 (B),
# 0 (C),
# 8 (Aflat).

# Sample output:
#[[1.8835162e-05 6.5159821e-04 4.9213314e-04 3.6079463e-04 7.0233677e-06
# 5.9515616e-04 2.8959711e-04 2.6659507e-03 4.4817248e-06 8.2715815e-06
# 9.9489611e-01 1.0026564e-05]
# [1.3361772e-04 1.3983199e-04 3.1511837e-03 2.7331302e-03 2.9068270e-03
# 4.3225122e-05 4.2098183e-03 2.8123530e-03 2.4182438e-03 4.3687625e-05
# 1.7295804e-04 9.8123521e-01]
# [9.8405254e-01 2.0168149e-05 1.2195716e-04 4.7840210e-04 4.8785005e-04
# 1.0949689e-02 1.5092465e-06 6.0027017e-04 2.5012414e-03 7.3335716e-04
# 3.4297787e-05 1.8673098e-05]
# [3.5973822e-03 7.4314848e-03 1.2533701e-04 3.5739841e-03 4.4565611e-03
# 3.3669064e-03 1.3955924e-04 1.1770831e-04 9.7345400e-01 2.1218757e-04
# 2.5890977e-04 3.2659373e-03]]
print(model.predict(X_validate))

# model.save_weights('model.hdf5')
# with open('model.json', 'w') as f:
#     f.write(model.to_json())

# print(X_validate[0])
