function cleanAllTimers()
    try
        stop(timerfindall)
        delete(timerfindall)
    catch
    end
end

