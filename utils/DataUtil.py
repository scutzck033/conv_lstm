class DataUtil:

    @staticmethod
    def getData(rawdata,startpoint,endpoint,n_hours,moving_len = 0):

        dateColumn = rawdata[:,0]
        startIndex = 0
        for i in range(dateColumn.shape[0]):
            if(cmp(startpoint,dateColumn[i]) == 0):
                startIndex = i
                break

        endIndex = 0
        for i in range(dateColumn.shape[0]):
            if(cmp(endpoint,dateColumn[i]) == 0):
                endIndex = i
                break

        len = (endIndex - startIndex)/n_hours + 1 - moving_len
        return rawdata[startIndex + moving_len * n_hours : endIndex+n_hours],len

