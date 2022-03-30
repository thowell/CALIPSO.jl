function positive_check(w) 
    for wi in w 
        if wi <= 0.0 
            return false 
        end
    end
    return true 
end