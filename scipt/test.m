function test1(mang)
    v = isStop(mang);
    disp(v);
end


function rs = isStop(mang)
    if size(mang, 2) == 1
        rs = true;
    elseif mang(1) > mang(2) && isStop(mang(2:end))
        rs = true;
    else
        rs = false;
    end
end