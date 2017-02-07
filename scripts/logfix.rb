require 'pocolog'
require 'pry'
require 'optparse'

input_file  = nil
output_file = nil
first_index = nil
last_index  = nil

parser = OptionParser.new do |opt|

    opt.banner = "Usage: logfix --file=FILENAME [--first_index=N] [--last_index=N]"

    opt.on('-i', '--input-file=FILENAME', 'The input filename') do |value|
        input_file = value
    end

    opt.on('-o', '--output-file=FILENAME', 'The output filename') do |value|
        output_file = value
    end

    opt.on('--first-index=N', 'First sample index') do |value|
        first_index = Integer(value)
    end

    opt.on('--last-index=N', 'Last sample index') do |value|
        last_index = Integer(value)
    end

    opt.on('-h', '--help', 'help') do
        puts opt
        exit
    end
end

parser.parse(ARGV)

if !input_file
    puts parser
    puts "Argument error: informe the log input filename."
    exit
end

input_logfile = Pocolog::Logfiles.open(input_file)

output_file = if !output_file then File.expand_path("output", File.dirname(__FILE__)) else output_file end
first_index = if !first_index then 0 else first_index end

output_logfile = Pocolog::Logfiles.create(output_file)
output_logfile.compress = false

input_logfile.streams.each do |input_stream|
    output_stream = output_logfile.stream(input_stream.name, input_stream.type, true)
    last_index = if !last_index || last_index > input_stream.size then input_stream.size else last_index end
    puts "input_stream: #{input_stream.name} size: #{input_stream.size}"
    input_stream.copy_to(first_index, last_index, output_stream)
end

input_logfile.close
output_logfile.close
