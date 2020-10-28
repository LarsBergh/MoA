model = Sequential()
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.15))
model.add(Dense(64, activation='elu')) 
model.add(Dropout(0.15))
model.add(Dense(64, activation='elu')) 
model.add(Dense(206, activation='softmax')) 
opti = SGD(lr=0.05, momentum=0.98)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=4, epochs=25, validation_data=(X_val, y_val), callbacks=[early_stop])

# %%
after_lis = []

y_pred = model.predict(X_test)
bce_before = BinaryCrossentropy()
bce_before = bce_before(y_test, y_pred).numpy()
print("BCE before", bce_before)

for i, percentage in enumerate(range(-10, 15)):
     
        
    #Loop over rows in y_df
    for i in range(y_pred.shape[0]):
        #Print first 2 rows of y_predicted and y_true (y_test)
        #print(y_pred[i])
        #print(y_test.iloc[i, :].values)

        #Find max value per row
        max_val = y_pred[i][y_pred[i].argmax()]

        #Find cutoff value 5% under argmax per row
        per_diff = percentage / 100
        cutoff = max_val - (max_val*per_diff) 
        #print(max_val, max_diff, cutoff)
        
        #set rows lower than cutoff to 0 and higher than cutoff to argmax value
        row = np.array(y_pred[i])
        #row[row <= cutoff] = 0
        row[row > cutoff] = max_val
        #y_pred[i][y_pred[i].argmax()] = 1

        #Change y_pred values
        y_pred[i] = row
        

    bce = BinaryCrossentropy()
    bce = bce(y_test, y_pred).numpy()
    after_lis.append([bce_before, bce])

print(pd.DataFrame(after_lis, columns=["before", "after"]))
#%%