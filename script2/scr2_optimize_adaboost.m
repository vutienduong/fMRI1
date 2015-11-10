function [result_table, max_acc, num_ite, num_feature] = scr2_optimize_adaboost(range_ite, step_ite, range_feature, step_feature)
	min_acc_gap = 0.0005;
	max_acc = 0;
	break_var = false;
	for num_ite=range_ite(1):step_ite:range_ite(2)
		for num_feature=range_feature(1):step_feature:range_feature(2)
			acc = scr2_run_ada(num_feature, num_ite);
			if acc > max_acc 
				if max_acc ~= 0
					break_var = true;
					break
				else
					max_acc = acc
				end
			else
				result_table{1} = [num_feature num_ite acc];
			end
		end
		if break_var
			break
		end
	end
	max_acc = acc;