## ------------------------------------------------------------------------
##
## Prepare learning data for model fitting.
## 
## Author: Maarten van der Velde
##
## Last updated: 2022-03-14
##
## ------------------------------------------------------------------------


library(data.table)
library(dplyr)




# Load data ---------------------------------------------------------------

load(file.path("data", "cogpsych_data_anon_2017.rda"))
cog_17 <- data

load(file.path("data", "cogpsych_data_anon_2018.rda"))
cog_18 <- data

cog <- rbind(cog_17, cog_18)
setDT(cog)

rm(data, exam.item, grades, study_time, cog_17, cog_18)




# Format columns ----------------------------------------------------------

d_all <- cog[, .(user = User,
                 course = case_when(
                   Course == "(17/18)) Cognitieve psychologie" ~ "17_18_NL",
                   Course == "(17/18)) Cognitive Psychology"   ~ "17_18_EN",
                   Course == "(18/19)) Cognitive Psychology"   ~ "18_19_EN"
                 ),
                 session,
                 time = Time,
                 fact = sub("_", "", factId, fixed = TRUE),
                 choices = Number.Of.Alternatives,
                 presentation_start_time = presentationStartTime / 1000,
                 rt = reactionTime,
                 correct
)]

d_all[, user := paste(course, user, sep = "_")]
d_all[, course := NULL]

# Add repetition counter
setorder(d_all, user, presentation_start_time)
d_all[, repetition := 1:.N, by = .(user, fact)]

# Add session boundary information
d_all[, time_since_session_start := presentation_start_time - min(presentation_start_time), by = .(user, session)]
d_all[, time_until_session_end := max(presentation_start_time) - presentation_start_time, by = .(user, session)]

# Distinguish interval types
d_all[, interval_type := ifelse(session == shift(session), "within", "between"), by = .(user, fact)]

# Align sequence start times
d_all[, presentation_start_time := (presentation_start_time - min(presentation_start_time)), by = .(user, fact)]




# Filter data -------------------------------------------------------------

# Only keep open-answer questions
d <- d_all[choices == 1,]

# Remove impossible RTs
# Computer clock adjustments during the trial can cause the trial to end up with a negative RT.
# If that occurs somewhere in a sequence, remove the entire sequence
d <- d[, .SD[min(c(rt, Inf), na.rm = T) > 0], by = .(user, fact)] # Pad with Inf in case all RTs are NA

# Only keep trials from sequences with at least 3 presentations in the first session
d_count <- d[, .(trials = .N), by = .(user, fact, session)]
d_keep <- d_count[, .(at_least_3 = trials[session == min(session)] >= 3), by = .(user, fact)][at_least_3 == TRUE, .(user, fact)]

# Only keep trials from sequences with at least one presentation in a later session
d_keep <- d_count[d_keep, on = .(user, fact)][
  , .(session = session[1:2]), by = .(user, fact)][
    , .(session = session[!any(is.na(session))]), by = .(user, fact)]

d_seq <- d[d_keep, on = .(user, fact, session)]

# Only keep the first trial in the second session (which follows a between-session interval)
d_seq <- d_seq[, .SD[session == min(session) | interval_type == "between"], by = .(user, fact)]




# Export data -------------------------------------------------------------

# Select columns
d_f <- d_seq[, .(id = paste(user, fact, sep = "_"),
                 user,
                 fact,
                 time,
                 presentation_start_time,
                 time_since_session_start,
                 time_until_session_end,
                 correct,
                 rt)]

# Decompose intervals into time within session and time between sessions
d_f[, time_between := (presentation_start_time[.N] - time_since_session_start[.N]) - (presentation_start_time[.N - 1] + time_until_session_end[.N - 1]), by = .(id)]
d_f[, time_within := time_since_session_start[.N] + time_until_session_end, by = .(id)]


fwrite(d_f, file.path("data", "cogpsych_data_formatted.csv"))
