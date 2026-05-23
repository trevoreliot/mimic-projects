select pat.subject_id,
       count(distinct(hadm_id)) as counts
from mimiciii.patients pat
left join mimiciii.admissions adm on pat.subject_id = adm.subject_id
group by pat.subject_id
order by counts desc
