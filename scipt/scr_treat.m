function rs = scr_treat(dt, expect)
    max_row = size(dt,1);
    for index=1:max_row
        m = dt(index, 1:end-1);
        expect = dt(index,end);
        min_expect = expect - 0.01;
        max_expect = expect + 0.01;
        max = size(m,2);
        for i=1:max - 5
            for j=i+1:max
                for k=j+1:max
                    for l=k+1:max
                        for mi=l+1:max
                            for n=mi+1:max
                                tempSum = mean([m(i) m(j) m(k) m(l) m(mi) m(n)]);
                                if  tempSum >= min_expect && tempSum <= max_expect
                                    rs = [i j k l mi n];
                                    return;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end