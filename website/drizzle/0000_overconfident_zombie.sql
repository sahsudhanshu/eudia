CREATE TABLE `citations` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`document_id` integer NOT NULL,
	`title` text NOT NULL,
	`x` real NOT NULL,
	`y` real NOT NULL,
	`citations` integer NOT NULL,
	`year` integer NOT NULL,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL,
	FOREIGN KEY (`document_id`) REFERENCES `documents`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `documents` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`title` text NOT NULL,
	`file_url` text NOT NULL,
	`file_size` integer NOT NULL,
	`upload_date` text NOT NULL,
	`status` text DEFAULT 'processing' NOT NULL,
	`user_id` text,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL
);
