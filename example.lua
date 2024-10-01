local LSTM = require("lstm")
local model_path = "model"

local function create_vocabulary(filename)
    local file = io.open(filename, "r")
    if not file then
        error("Could not open file: " .. filename)
    end

    local vocab = {}
    local vocab_size = 0
    local char_to_index = {}
    local index_to_char = {}

    while true do
        local char = file:read(1)
        if not char then break end

        if not char_to_index[char] then
            vocab_size = vocab_size + 1
            char_to_index[char] = vocab_size
            index_to_char[vocab_size] = char
        end
    end

    file:close()

    return {
        size = vocab_size,
        char_to_index = char_to_index,
        index_to_char = index_to_char
    }
end

local function create_sequences(filename, vocab, sequence_length)
    local file = io.open(filename, "r")
    if not file then
        error("Could not open file: " .. filename)
    end

    local sequences = {}
    local current_sequence = {}

    while true do
        local char = file:read(1)
        if not char then break end

        local index = vocab.char_to_index[char]
        if index then
            table.insert(current_sequence, index)

            if #current_sequence == sequence_length + 1 then
                local input_seq = { unpack(current_sequence, 1, sequence_length) }
                local target_seq = { unpack(current_sequence, 2, sequence_length + 1) }
                table.insert(sequences, { input = input_seq, target = target_seq })
                table.remove(current_sequence, 1)
            end
        end
    end

    file:close()

    return sequences
end

-- Training function
local function train_lstm(lstm, sequences, num_epochs, learning_rate)
    for epoch = 1, num_epochs do
        local total_loss = 0
        for i, sequence in ipairs(sequences) do
            lstm:forward(sequence.input)
            local loss = lstm:backward(sequence.input, sequence.target, learning_rate)
            total_loss = total_loss + loss

            if i % 100 == 0 then
                print(string.format("Epoch %d, Sequence %d/%d, Loss: %.4f", epoch, i, #sequences, loss))
            end
        end
        print(string.format("Epoch %d completed, Average Loss: %.4f", epoch, total_loss / #sequences))

        if epoch % 20 == 0 then
            lstm:save(model_path)
        end
    end
end

-- Example usage
local filename = "test.txt"
local sequence_length = 5
local embedding_size = 10
local hidden_size = 50
local num_layers = 2
local num_epochs = 100
local learning_rate = 0.001

-- Create vocabulary from file
local vocab = create_vocabulary(filename)

-- Create LSTM network
local lstm = LSTM.new(vocab.size, embedding_size, hidden_size, num_layers)

-- Create input sequences from file
local sequences = create_sequences(filename, vocab, sequence_length)

-- Train the LSTM
train_lstm(lstm, sequences, num_epochs, learning_rate)

-- Generate text
local seed_sequence = sequences[1].input         -- Use the first sequence as seed
local generated_indices = lstm:generate(seed_sequence, 200)

print("\nGenerated text:")
for i, index in ipairs(generated_indices) do
    io.write(vocab.index_to_char[index])
end
print()
