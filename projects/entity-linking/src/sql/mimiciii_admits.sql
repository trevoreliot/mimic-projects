select admits.subject_id,
	pt.gender,
	pt.dob, 
	hadm_id,
	admits.admission_location,
	admits.discharge_location,
	admits.insurance,
	admits.admission_type 
from mimiciii.admissions admits
left join mimiciii.patients pt on admits.subject_id = pt.subject_id 