#------------------------ YET TO SUBMIT ------------------------#
#Loss on X_val, y_val = 0.0188
#Loss on submit = 
#Leaderboard approx position = 
#Multilabel classification baseline model
l1_reg = 0.00000001
dropout_val = 0.2

model = Sequential()
model.add(Dense(128, activation='elu', activity_regularizer=L1L2(l1_reg)))
model.add(Dropout(dropout_val))
model.add(Dense(128, activation='elu', activity_regularizer=L1L2(l1_reg))) 
model.add(Dropout(dropout_val))
model.add(Dense(128, activation='elu', activity_regularizer=L1L2(l1_reg))) 
model.add(Dense(207, activation='sigmoid')) 
opti = SGD(lr=0.05, momentum=0.99)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=8, epochs=50)

#Get validation loss/acc
results = model.evaluate(X_val, y_val, batch_size=1)

#Predict on test set to get final results
y_pred = model.predict(X_test)

#Predict values for submit
y_submit = model.predict(X_submit)



#Loss on X_val, y_val = 0.0188
#Loss on submit = 
#Leaderboard approx position = 
#Multilabel classification baseline model
l1_reg = 0.0000001
dropout_val = 0.15

model = Sequential()
model.add(Dense(64, activation='elu', activity_regularizer=L1L2(l1_reg)))
model.add(Dropout(dropout_val))
model.add(Dense(64, activation='elu', activity_regularizer=L1L2(l1_reg))) 
model.add(Dropout(dropout_val))
model.add(Dense(64, activation='elu', activity_regularizer=L1L2(l1_reg))) 
model.add(Dense(207, activation='sigmoid')) 
opti = SGD(lr=0.05, momentum=0.98)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=8, epochs=50)

#Get validation loss/acc
results = model.evaluate(X_val, y_val, batch_size=1)

#Predict on test set to get final results
y_pred = model.predict(X_test)

#Predict values for submit
y_submit = model.predict(X_submit)


#Loss on X_val, y_val = 0.0189
#Loss on submit = 0,02058
#Leaderboard approx position = 
#Multilabel classification baseline model
model = Sequential()
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.15))
model.add(Dense(64, activation='elu')) 
model.add(Dropout(0.15))
model.add(Dense(64, activation='elu')) 
model.add(Dense(207, activation='softmax')) 
opti = SGD(lr=0.05, momentum=0.98)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=4, epochs=50)

#Get validation loss/acc
results = model.evaluate(X_val, y_val, batch_size=1)

#Predict on test set to get final results
y_pred = model.predict(X_test)

#Predict values for submit
y_submit = model.predict(X_submit)





#Loss on X_val, y_val = 0.0190
#Loss on submit = 
#Leaderboard approx position = 
#Multilabel classification baseline model
model = Sequential()
model.add(Dense(32, activation='elu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='elu')) 
model.add(Dropout(0.2))
model.add(Dense(207, activation='softmax')) 
opti = SGD(lr=0.1, momentum=0.95)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=4, epochs=50)

#Get validation loss/acc
results = model.evaluate(X_val, y_val, batch_size=1)

#Predict on test set to get final results
y_pred = model.predict(X_test)

#Predict values for submit
y_submit = model.predict(X_submit)





#Loss on X_val, y_val = 0.0195
#Loss on submit = 
#Leaderboard approx position = 
#Multilabel classification baseline model
model = Sequential()
model.add(Dense(32, activation='elu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='elu')) 
model.add(Dropout(0.2))
model.add(Dense(207, activation='softmax')) 
opti = SGD(lr=0.1, momentum=0.95)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=8, epochs=40)

#Get validation loss/acc
results = model.evaluate(X_val, y_val, batch_size=1)

#Predict on test set to get final results
y_pred = model.predict(X_test)

#Predict values for submit
y_submit = model.predict(X_submit)




#Loss on X_val, y_val = 0.0201
#Loss on submit = 
#Leaderboard approx position = 
#Multilabel classification baseline model
model = Sequential()
model.add(Dense(32, activation='elu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='elu')) 
model.add(Dropout(0.2))
model.add(Dense(207, activation='softmax')) 
opti = SGD(lr=0.1, momentum=0.95)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=8, epochs=20)

#Get validation loss/acc
results = model.evaluate(X_val, y_val, batch_size=1)

#Predict on test set to get final results
y_pred = model.predict(X_test)

#Predict values for submit
y_submit = model.predict(X_submit)



#------------------------ Submitted ------------------------#
#Loss on X_val, y_val = 0.0211
#Loss on submit = 0.02151
#Leaderboard approx position = 1673
#Multilabel classification baseline model
model = Sequential()
model.add(Dense(32, activation='elu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='elu')) 
model.add(Dropout(0.2))
model.add(Dense(207, activation='sigmoid')) 
opti = SGD(lr=0.1, momentum=0.95)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=8, epochs=20)

#Get validation loss/acc
results = model.evaluate(X_val, y_val, batch_size=8)