
model = Sequential()
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.15))
model.add(Dense(64, activation='elu')) 
model.add(Dropout(0.15))
model.add(Dense(64, activation='elu')) 
model.add(Dense(206, activation='softmax')) 
opti = SGD(lr=0.05, momentum=0.98)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=4, epochs=25, validation_data=(X_val, y_val))

#Get validation loss/acc
results = model.evaluate(X_test, y_test, batch_size=1)

#Predict on test set to get final results
y_pred = model.predict(X_test)

#Predict values for submit
y_submit = model.predict(X_submit)


def create_model(X_train, X_val, y_train, y_val, lay, acti_hid, acti_out, neur, drop, epo, opti):
    #Print model parameters
    print("Creating model with:")
    print("Activation hidden: ", acti_hid)
    print("Activation output: ", acti_out)
    print("Hidden layer count: ", lay)
    print("Neuron count per layer: ", neur)
    print("Dropout value: ", drop)
    print("Epoch count: ", epo)
    print("Optimizer: ", opti)

    
    #Create model
    model = Sequential()
    
    #Create layers based on count with specified activations and dropouts
    for l in range(0,lay):
        model.add(BatchNormalization())
        model.add(Dense(neur, activation=acti_hid, activity_regularizer=L1L2(L1_REG)))
        
        #Add dropout except for last layer
        if l != lay - 1:
            model.add(Dropout(drop))

    #Add output layer
    model.add(Dense(206, activation=acti_out)) 

    #Define optimizer and loss
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
    
    #Define callbacks
    hist = History()
    early_stop = EarlyStopping(monitor='val_loss', patience=2, mode='auto')

    #Fit and return model and loss history
    model.fit(X_train, y_train, batch_size=64, epochs=epo, validation_data=(X_val, y_val), callbacks=[early_stop, hist])
    print(model.summary())
    return model, hist