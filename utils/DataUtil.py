class DataUtil:

    @staticmethod
    def getData(rawdata,startpoint,endpoint,n_hours,moving_len = 0):# in this case,
                                                    # moving_len is used to get the startDate for prediction

        dateColumn = rawdata[:,0]

        startIndex = 0
        for i in range(dateColumn.shape[0]):
            if(cmp(startpoint,dateColumn[i]) == 0):
                startIndex = i
                break
        print (startIndex)
        endIndex = 0
        for i in range(dateColumn.shape[0]):
            if(cmp(endpoint,dateColumn[i]) == 0):
                endIndex = i
                break
        print (endIndex)

        overflow = (endIndex - startIndex + 1)%n_hours
        if(overflow != 0):
            endIndex += n_hours - overflow
            print ("The endpoint %s is not supported ,with endpoint %s replaced"%(dateColumn[endIndex-n_hours+overflow],dateColumn[endIndex]))

        len = (endIndex - startIndex)/n_hours + 1 - moving_len
        # return rawdata[startIndex + moving_len * n_hours : endIndex+n_hours],len #-- in the case of only date given
        return rawdata[startIndex + moving_len * n_hours : endIndex+1],len #-- in the case of datetime given
