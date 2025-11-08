import { db } from '@/db';
import { documents } from '@/db/schema';

async function main() {
    const now = new Date();
    
    const sampleDocuments = [
        {
            title: 'Brown v. Board of Education Analysis',
            fileUrl: '/documents/brown-v-board-of-education-analysis.pdf',
            fileSize: 1250000,
            uploadDate: new Date(now.getTime() - 25 * 24 * 60 * 60 * 1000).toISOString(),
            status: 'completed',
            userId: null,
            createdAt: new Date(now.getTime() - 25 * 24 * 60 * 60 * 1000).toISOString(),
            updatedAt: new Date(now.getTime() - 25 * 24 * 60 * 60 * 1000).toISOString(),
        },
        {
            title: 'Miranda Rights Legal Framework',
            fileUrl: '/documents/miranda-rights-legal-framework.pdf',
            fileSize: 890000,
            uploadDate: new Date(now.getTime() - 15 * 24 * 60 * 60 * 1000).toISOString(),
            status: 'completed',
            userId: null,
            createdAt: new Date(now.getTime() - 15 * 24 * 60 * 60 * 1000).toISOString(),
            updatedAt: new Date(now.getTime() - 15 * 24 * 60 * 60 * 1000).toISOString(),
        },
        {
            title: 'Roe v. Wade Historical Context',
            fileUrl: '/documents/roe-v-wade-historical-context.pdf',
            fileSize: 1750000,
            uploadDate: new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
            status: 'processing',
            userId: null,
            createdAt: new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
            updatedAt: new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
        },
    ];

    await db.insert(documents).values(sampleDocuments);
    
    console.log('✅ Documents seeder completed successfully');
}

main().catch((error) => {
    console.error('❌ Seeder failed:', error);
});