package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"github.com/RB387/wolt-ai-agents-talk/internal"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
)

const systemPrompt = `You are a helpful assistant with access to tools. Use the provided tools when needed to answer the user's question.`

func ping(website string) string {
	if !strings.HasPrefix(website, "https://") && !strings.HasPrefix(website, "http://") {
		website = "https://" + website
	}
	start := time.Now()
	resp, err := http.Get(website)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	defer resp.Body.Close()
	return fmt.Sprintf("%.2f seconds", time.Since(start).Seconds())
}

func bash(command string) string {
	cmd := exec.Command("bash", "-c", command)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	return strings.TrimSpace(string(output))
}

func webSearch(query string) string {
	searchClient := internal.NewSearchClient()
	results, err := searchClient.Search(query)
	if err != nil {
		return fmt.Sprintf("Error performing web search: %v", err)
	}
	if len(results) == 0 {
		return "No results found"
	}
	jsonResults, err := json.Marshal(results)
	if err != nil {
		return fmt.Sprintf("Error marshaling results: %v", err)
	}
	return string(jsonResults)
}

func scrape(url string) string {
	scraperClient := internal.NewScraperClient()
	content, err := scraperClient.Scrape(url)
	if err != nil {
		return fmt.Sprintf("Error scraping content: %v", err)
	}
	return content
}

func toolsDef() []openai.ChatCompletionToolParam {
	return []openai.ChatCompletionToolParam{
		{
			Function: shared.FunctionDefinitionParam{
				Name:        "ping",
				Description: openai.String("Does a ping/HTTP request to the given URL and returns the response time in seconds. Pass a hostname or URL, e.g. wolt.com or https://wolt.com"),
				Parameters: shared.FunctionParameters{
					"type": "object",
					"properties": map[string]any{
						"website": map[string]string{"type": "string", "description": "Hostname or URL to measure response time"},
					},
					"required": []string{"website"},
				},
			},
		},
		{
			Function: shared.FunctionDefinitionParam{
				Name:        "bash",
				Description: openai.String("Runs a bash command and returns the output"),
				Parameters: shared.FunctionParameters{
					"type": "object",
					"properties": map[string]any{
						"command": map[string]string{"type": "string", "description": "Bash command to execute"},
					},
					"required": []string{"command"},
				},
			},
		},
		{
			Function: shared.FunctionDefinitionParam{
				Name:        "web_search",
				Description: openai.String("Searches the web and returns JSON with URLs of search results. Use for factual or current information."),
				Parameters: shared.FunctionParameters{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]string{"type": "string", "description": "Search query"},
					},
					"required": []string{"query"},
				},
			},
		},
		{
			Function: shared.FunctionDefinitionParam{
				Name:        "scrape",
				Description: openai.String("Fetches and returns the text content of the given URL"),
				Parameters: shared.FunctionParameters{
					"type": "object",
					"properties": map[string]any{
						"url": map[string]string{"type": "string", "description": "URL to scrape"},
					},
					"required": []string{"url"},
				},
			},
		},
	}
}

func runTool(name, argsJSON string) string {
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return fmt.Sprintf("Error parsing arguments: %v", err)
	}
	switch name {
	case "ping":
		return ping(args["website"].(string))
	case "bash":
		return bash(args["command"].(string))
	case "web_search":
		return webSearch(args["query"].(string))
	case "scrape":
		return scrape(args["url"].(string))
	default:
		return "Unknown tool: " + name
	}
}

func runAgentLoop(client openai.Client, userQuery string, maxIter int) {
	ctx := context.Background()
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(systemPrompt),
		openai.UserMessage(userQuery),
	}
	tools := toolsDef()

	for i := 0; i < maxIter; i++ {
		fmt.Printf("Loop: %d\n", i+1)

		params := openai.ChatCompletionNewParams{
			Messages: messages,
			Model:    shared.ChatModelGPT4o,
			Tools:    tools,
		}

		completion, err := client.Chat.Completions.New(ctx, params)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			return
		}

		msg := completion.Choices[0].Message
		if msg.Content != "" {
			fmt.Println(msg.Content)
		}

		if len(msg.ToolCalls) == 0 {
			fmt.Println("No more tool calls, agent is done.")
			break
		}

		messages = append(messages, msg.ToParam())
		for _, tc := range msg.ToolCalls {
			name := tc.Function.Name
			args := tc.Function.Arguments
			fmt.Printf("Tool call: %s %s\n", name, args)
			result := runTool(name, args)
			fmt.Printf("Result: %s\n", result)
			messages = append(messages, openai.ToolMessage(result, tc.ID))
		}
	}
}

func main() {
	client := internal.NewOpenAIClient()

	fmt.Println("=== Query 1: Response time for wolt.com ===")
	runAgentLoop(client, "What's the response time for wolt.com?", 5)

	fmt.Println("\n=== Query 2: Go version ===")
	runAgentLoop(client, "What version of Golang is installed on this machine?", 5)

	fmt.Println("\n=== Query 3: What is the weather in Helsinki today ===")
	runAgentLoop(client, "What is the weather in Helsinki today (in Celsius)? Also print time when the weather was checked. Preferably from accuweather", 5)
}
