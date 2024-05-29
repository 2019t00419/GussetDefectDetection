import numpy as np

def fill_coordinates(contour_array):
    insert_index=0
    last_x, last_y = contour_array[len(contour_array) - 1][0]
    new_array =contour_array
    for coordinates in contour_array:
        next_x,next_y=coordinates[0]
        if(next_x-last_x < -1 or next_x-last_x > 1) and (next_y - last_y ==0):
            if next_x > last_x:
                new_x = next_x-1
                new_y = last_y
                count=0
                while new_x > last_x:
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x-1
                    count=count+1
                insert_index=insert_index+count    
                if insert_index>len(new_array)-1:
                    break                        
                            
            else:
                new_x = next_x+1
                new_y = next_y
                count=0
                while new_x < last_x:
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x+1 
                    count=count+1
                insert_index=insert_index+count     
                if insert_index>len(new_array)-1:
                    break                                    
        #vertical case
        elif(next_y-last_y < -1 or next_y-last_y > 1) and (next_x - last_x == 0):
            if next_y > last_y:
                new_y = next_y-1
                new_x = last_x
                count=0
                while new_y > last_y:
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_y = new_y-1
                    count=count+1
                insert_index=insert_index+count     
                if insert_index>len(new_array)-1:
                    break                                
            else:
                new_x = next_x
                new_y = next_y+1
                count=0
                while new_y < last_y:
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_y = new_y+1
                    count=count+1
                insert_index=insert_index+count  
                if insert_index>len(new_array)-1:
                    break                       
                            
        #angled case    
        elif(next_x-last_x < -1 or next_x-last_x > 1) and (next_y-last_y < -1 or next_y-last_y > 1):

            #print(str(next_x)+","+str(next_y))
            #print(str(last_x)+","+str(last_y))
            line_gradient = ( (next_y - last_y)/(next_x-last_x) )
            line_intercept = ( next_y-(line_gradient*next_x))

            if next_x > last_x:
                    
                new_y = next_y
                new_x = next_x-1
                count=0
                while new_x > last_x:
                    new_y = (line_gradient*new_x)+line_intercept
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x-1
                    count=count+1
                insert_index=insert_index+count  
                if insert_index>len(new_array)-1:
                    break                       
                            
            else:
                new_x = next_x+1
                new_y = next_y
                count=0
                while new_x < last_x:
                    new_y = (line_gradient*new_x)+line_intercept
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x+1  
                    count=count+1  
                insert_index=insert_index+count 
                if insert_index>len(new_array)-1:
                    break                        
        #print(str(next_x)+","+str(next_y))
        #print(str(last_x)+","+str(last_y))
            
        insert_index=insert_index+1    
        last_y = next_y
        last_x = next_x
    return new_array
