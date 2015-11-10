function testF(mang)
max_len = size(mang,2);
max_loop = 25;
    while max_loop 
        if isStop(mang)
            disp('end');
            return
        else
            %disp('not stop');
            for j = max_len-1:-1:1
                if mang(j)<mang(j+1)
                    min_val = mang(j+1);
                    %disp(['not stop 2 :', min_val]);
                    min_idx = j +1;
                    for k = j: max_len
                        if mang(k) > mang(j) && mang(k) <= min_val
                            min_val = mang(k);
                            min_idx = k;
                        end
                    end
                    mang([j min_idx]) = mang([min_idx j]);
                    mang = [ mang(1:j) fliplr(mang(j+1:end))];
                    disp(mang);
                    break;
                end
            end
        end
        max_loop = max_loop-1;
    end
end


function rs = isStop(mang)
    disp(mang);
    if size(mang, 2) == 1
        rs = true;
    elseif mang(1) > mang(2) && isStop(mang(2:end))
        rs = true;
    else
        rs = false;
    end
end
