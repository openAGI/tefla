# Set the log file name, using append mode

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/logger.py#L28 target="_blank"><b>tefla.core.logger.setFileHandler</b></a></span>  (filename,  mode='a')</span>

<h3>Args</h3>


 - **filename**: log file name
 - e.g. tefla.log
 - **mode**: file writing mode, append/over write if exists else start new file
 - e.g. a string, 'a' or 'w'

 ---------- 

# set the verbosity level of logging

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/logger.py#L80 target="_blank"><b>tefla.core.logger.setVerbosity</b></a></span>  (verbosity=0)</span>

<h3>Args</h3>


 - **verbosity**: set the verbosity level using an integer {0, 1, 2, 3, 4} 
 - e.g. verbosity=0, imply DEBUG logging, it logs all level of logs
 verbosity=1, imply INFO logging
 verbosity=2, imply WARN logging
 verbosity=3, imply ERROR logging
 verbosity=4, imply FATAL logging, it logs only the lowest FATAL level

 ---------- 

# Logs the Highest level DEBUG logging, it logs all level

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/logger.py#L113 target="_blank"><b>tefla.core.logger.debug</b></a></span>  (msg,  *args,  **kwargs)</span>

<h3>Args</h3>

 
 - **msg**: the message to log

 ---------- 

# Logs the level INFO logging, it logs all LEVEL BELOW INFO

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/logger.py#L123 target="_blank"><b>tefla.core.logger.info</b></a></span>  (msg,  *args,  **kwargs)</span>

<h3>Args</h3>

 
 - **msg**: the message to log

 ---------- 

# Logs the WARN logging, it logs all level BELOW WARN

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/logger.py#L133 target="_blank"><b>tefla.core.logger.warn</b></a></span>  (msg,  *args,  **kwargs)</span>

<h3>Args</h3>

 
 - **msg**: the message to log

 ---------- 

# Logs the level ERROR logging, it logs level ERROR  and FATAL

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/logger.py#L143 target="_blank"><b>tefla.core.logger.error</b></a></span>  (msg,  *args,  **kwargs)</span>

<h3>Args</h3>

 
 - **msg**: the message to log

 ---------- 

# Logs thE level FATAL logging, it logs only FATAL

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/logger.py#L153 target="_blank"><b>tefla.core.logger.fatal</b></a></span>  (msg,  *args,  **kwargs)</span>

<h3>Args</h3>

 
 - **msg**: the message to log

 ---------- 

