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
        # print (startIndex)
        endIndex = 0
        for i in range(dateColumn.shape[0]):
            if(cmp(endpoint,dateColumn[i]) == 0):
                endIndex = i
                break
        # print (endIndex)

        startpoint_overflow = startIndex % n_hours
        if (startpoint_overflow != 0):
            startIndex -= startpoint_overflow
            print ("The startpoint %s is not supported ,with startpoint %s replaced" % (
            dateColumn[startIndex + startpoint_overflow], dateColumn[startIndex]))


        endpoint_overflow = (endIndex - startIndex + 1)%n_hours
        if(endpoint_overflow != 0):
            endIndex += n_hours - endpoint_overflow
            print ("The endpoint %s is not supported ,with endpoint %s replaced"
                   %(dateColumn[endIndex-n_hours+endpoint_overflow],dateColumn[endIndex]))

        len = (endIndex - startIndex)/n_hours + 1 - moving_len
        # return rawdata[startIndex + moving_len * n_hours : endIndex+n_hours],len #-- in the case of only date given
        return rawdata[startIndex + moving_len * n_hours : endIndex+1],len #-- in the case of datetime given

    @staticmethod
    def tendencyMatch(array1,array2):
        num_match = 0
        num_mismatch = 0
        length = array1.shape[0]
        print (array1)
        print (array2)
        print (length)

        for i in range(length-1):
            if(((array1[i+1]-array1[i])*(array2[i+1]-array2[i]))>0):
                num_match+=1
            else:
                num_mismatch+=1
        print ("num_match:%d"%num_match)
        print ("num_mismatch:%d"%num_mismatch)

        match_percent = float(num_match)/float(num_match+num_mismatch)
        # mismatch_percent = num_mismatch/(num_match+num_mismatch)

        return match_percent#,mismatch_percent

